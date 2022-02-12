import torch


class FGM():
    """
    Fast Gradient Method, ICLR-2017

    Examples
    --------
    >>> fgm.attack(epsilon, 'word_embeddings')
    >>> loss_adv = model(**batch).loss
    >>> loss_adv = loss_adv / args.gradient_accumulation_steps
    >>> accelerator.backward(loss_adv)
    >>> fgm.restore('word_embeddings')
    >>> optimizer.step()

    References
    ----------
    1. https://zhuanlan.zhihu.com/p/91269728
    """
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    """
    Projected Gradient Descent, ICLR-2018 

    Examples
    --------
    >>> pgd.backup_grad()
    >>> for t in range(K):
    >>>     pgd.attack(emb_name='word_embeddings', is_first_attack=(t==0))
    >>>     if t != K-1:
    >>>         optimizer.zero_grad()
    >>>     else:
    >>>         pgd.restore_grad()
    >>>     loss_adv = model(**batch).loss
    >>>     loss_adv = loss_adv / args.gradient_accumulation_steps
    >>>     accelerator.backward(loss_adv)
    >>> pgd.restore(emb_name='word_embeddings')

    References
    ----------
    1. https://zhuanlan.zhihu.com/p/91269728
    """
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='emb.', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
        
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r
        
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()
    
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


class FreeLB(object):
    """
    Free Large-Batch, ICLR-2020

    Examples
    --------
    >>> # FreeLB training
    >>> freelb.backup_grad()
    >>> for t in range(K):
    >>>     freelb.attack(emb_name='word_embeddings', is_first_attack=(t==0))
    >>>     if t == 0:
    >>>         optimizer.zero_grad()
    >>>     loss_adv = model(**batch).loss
    >>>     loss_adv = loss_adv / K
    >>>     loss_adv = loss_adv / args.gradient_accumulation_steps  
    >>>     accelerator.backward(loss_adv)
    >>> freelb.restore_grad()
    >>> freelb.restore(emb_name='word_embeddings')
    >>> optimizer.step()

    >>> # Smart training (Smart Without Smoothness-Inducing Adversarial Regularization)
    >>> freelb.backup_grad()
    >>> for t in range(K):                    
    >>>     freelb.attack(emb_name='word_embeddings', is_first_attack=(t==0))
    >>>     if t == 0:
    >>>         optimizer.zero_grad()
    >>>     loss_adv,logits_adv = model(**batch)[:2]
    >>>     loss_adv = loss_adv / K 
    >>>     loss_adv = loss_adv / args.gradient_accumulation_steps
    >>>     accelerator.backward(loss_adv)
    >>> freelb.restore_grad()
    >>> ###Bregman Proximal Point Optimization (prevent the model from aggressive update)
    >>> loss_adv_last,logits_adv_last = model(**batch)[:2]
    >>> loss_kl = stable_kl(logits_adv_last, logits.detach(), reduce=False)
    >>> loss_kl = loss_kl / args.gradient_accumulation_steps
    >>> accelerator.backward(loss_kl * breg_miu)
    >>> freelb.restore(emb_name='word_embeddings')
    >>> optimizer.step()

    References
    ----------
    1. https://github.com/cuixuage/tf_bert_competitions/tree/7c39ae157de2c716c7c1eafbe67b143107008650
    """
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='emb.', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                norm = torch.norm(param.grad)
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()
    
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = param.grad + self.grad_backup[name]
