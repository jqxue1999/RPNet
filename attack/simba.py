import torch
import torch.nn.functional as F
import utils
import math


class SimBA:

    def __init__(self, model, dataset, image_size, sigma=0):
        self.model = model
        self.dataset = dataset
        self.image_size = image_size
        self.model.eval()
        self.sigma = sigma
        self.channel = 1 if dataset == "mnist" else 3

    def expand_vector(self, x, size):
        batch_size = x.size(0)
        x = x.view(-1, self.channel, size, size)
        z = torch.zeros(batch_size, self.channel, self.image_size, self.image_size)
        z[:, :, :size, :size] = x
        return z

    def normalize(self, x):
        return utils.apply_normalization(x, self.dataset, self.sigma)

    def get_probs(self, x, y, a1=0, a2=0):
        with torch.no_grad():
            input = x.clone()
            for i in range(input.size(0)):
                input[i, :, :] += a1 * torch.randn_like(input[i, :, :])
            output = self.model(self.normalize(input.cuda())).cpu()
            for i in range(output.size(0)):
                output[i, :] += a2 * torch.randn_like(output[i, :])
            probs = torch.index_select(output, 1, y)
        return torch.diag(probs)

    def get_preds(self, x):
        output = self.model(self.normalize(x.cuda())).cpu()
        _, preds = output.data.max(1)
        return preds

    # 20-line implementation of SimBA for single image input
    def simba_single(self, x, y, num_iters=10000, epsilon=0.2, targeted=False):
        n_dims = x.view(1, -1).size(1)
        perm = torch.randperm(n_dims)
        x = x.unsqueeze(0)
        last_prob = self.get_probs(x, y)
        for i in range(num_iters):
            diff = torch.zeros(n_dims)
            diff[perm[i]] = epsilon
            left_prob = self.get_probs((x - diff.view(x.size())).clamp(0, 1), y)
            if targeted != (left_prob < last_prob):
                x = (x - diff.view(x.size())).clamp(0, 1)
                last_prob = left_prob
            else:
                right_prob = self.get_probs((x + diff.view(x.size())).clamp(0, 1), y)
                if targeted != (right_prob < last_prob):
                    x = (x + diff.view(x.size())).clamp(0, 1)
                    last_prob = right_prob
            if i % 10 == 0:
                print(last_prob)
        return x.squeeze()

    # runs simba on a batch of images <images_batch> with true labels (for untargeted attack) or target labels
    # (for targeted attack) <labels_batch>
    def simba_batch(self, images_batch, labels_batch, max_iters, epsilon, linf_bound=0.0, targeted=False,
                    pixel_attack=False, seed=47, T=20, beta_range=[0.5, 1.5], max_average_queries=1000):
        res = {0: 0}
        l2 = {0: 0}
        utils.setup_seed(seed)
        batch_size = images_batch.size(0)
        image_size = images_batch.size(2)
        assert self.image_size == image_size
        # sample a random ordering for coordinates independently per batch element
        indices = utils.diagonal_order(image_size, self.channel)[:max_iters]
        expand_dims = image_size
        n_dims = self.channel * expand_dims * expand_dims
        x = torch.zeros(batch_size, n_dims)
        # logging tensors
        probs = torch.zeros(batch_size, max_iters)
        succs = torch.zeros(batch_size, max_iters)
        queries = torch.zeros(batch_size, max_iters)
        l2_norms = torch.zeros(batch_size, max_iters)
        linf_norms = torch.zeros(batch_size, max_iters)
        prev_probs = self.get_probs(images_batch, labels_batch)
        preds = self.get_preds(images_batch)
        if pixel_attack:
            trans = lambda z: z
        else:
            trans = lambda z: utils.block_idct(z, block_size=image_size, linf_bound=linf_bound)
        remaining_indices = torch.arange(0, batch_size).long()
        X, y = [], []
        t = 0
        for k in range(max_iters):
            if T != 0:
                epsilon_cur = ((beta_range[1] + beta_range[0]) / 2 + (beta_range[1] - beta_range[0]) / 2 * math.cos(math.pi * t / T)) * epsilon
                if t == T:
                    T, t = T * 1, 0
                else:
                    t += 1
                y.append(epsilon_cur)
            else:
                epsilon_cur = epsilon
            dim = indices[k]
            expanded = (images_batch[remaining_indices] + trans(
                self.expand_vector(x[remaining_indices], expand_dims))).clamp(0, 1)
            perturbation = trans(self.expand_vector(x, expand_dims))
            l2_norms[:, k] = perturbation.view(batch_size, -1).norm(2, 1)
            linf_norms[:, k] = perturbation.view(batch_size, -1).abs().max(1)[0]
            preds_next = self.get_preds(expanded)
            preds[remaining_indices] = preds_next
            if targeted:
                remaining = preds.ne(labels_batch)
            else:
                remaining = preds.eq(labels_batch)
            # check if all images are misclassified and stop early
            if remaining.sum() == 0:
                adv = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(0, 1)
                probs_k = self.get_probs(adv, labels_batch)
                probs[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
                succs[:, k:] = torch.ones(batch_size, max_iters - k)
                queries[:, k:] = torch.zeros(batch_size, max_iters - k)
                break
            remaining_indices = torch.arange(0, batch_size)[remaining].long()
            if k > 0:
                succs[:, k - 1] = ~remaining
            diff = torch.zeros(remaining.sum(), n_dims)
            diff[:, dim] = epsilon_cur
            left_vec = x[remaining_indices] - diff
            right_vec = x[remaining_indices] + diff
            # trying negative direction
            adv = (images_batch[remaining_indices] + trans(self.expand_vector(left_vec, expand_dims))).clamp(0, 1)
            left_probs = self.get_probs(adv, labels_batch[remaining_indices])
            queries_k = torch.zeros(batch_size)
            # increase query count for all images
            queries_k[remaining_indices] += 1
            if targeted:
                improved = left_probs.gt(prev_probs[remaining_indices])
            else:
                improved = left_probs.lt(prev_probs[remaining_indices])
            # only increase query count further by 1 for images that did not improve in adversarial loss
            if improved.sum() < remaining_indices.size(0):
                queries_k[remaining_indices[~improved]] += 1
            # try positive directions
            adv = (images_batch[remaining_indices] + trans(self.expand_vector(right_vec, expand_dims))).clamp(0, 1)
            right_probs = self.get_probs(adv, labels_batch[remaining_indices])
            if targeted:
                right_improved = right_probs.gt(torch.max(prev_probs[remaining_indices], left_probs))
            else:
                right_improved = right_probs.lt(torch.min(prev_probs[remaining_indices], left_probs))
            probs_k = prev_probs.clone()
            # update x depending on which direction improved
            if improved.sum() > 0:
                left_indices = remaining_indices[improved]
                left_mask_remaining = improved.unsqueeze(1).repeat(1, n_dims)
                x[left_indices] = left_vec[left_mask_remaining].view(-1, n_dims)
                probs_k[left_indices] = left_probs[improved]
            if right_improved.sum() > 0:
                right_indices = remaining_indices[right_improved]
                right_mask_remaining = right_improved.unsqueeze(1).repeat(1, n_dims)
                x[right_indices] = right_vec[right_mask_remaining].view(-1, n_dims)
                probs_k[right_indices] = right_probs[right_improved]
            probs[:, k] = probs_k
            queries[:, k] = queries_k
            prev_probs = probs[:, k]
            rate = float(1 - remaining.float().mean()) * 100
            l2_last = float(l2_norms[:, k].mean())
            queries_num = queries.sum(1).mean()
            X.append(queries_num)
            if (k + 1) % 10 == 0 or k == max_iters - 1:
                print('Iteration %d: queries = %.4f, prob = %.4f, remaining = %.4f' % (
                    k + 1, queries.sum(1).mean(), probs[:, k].mean(), remaining.float().mean()))
        expanded = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(0, 1)
        preds = self.get_preds(expanded)
        if targeted:
            remaining = preds.ne(labels_batch)
        else:
            remaining = preds.eq(labels_batch)
        succs[:, max_iters - 1] = ~remaining