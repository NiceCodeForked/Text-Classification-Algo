import torch
from torchmetrics import Precision, Recall


def accuracy_score(
    y_true, 
    y_pred, 
    bootstrap=False, 
    num_rounds=200, 
    ci=.95, 
    unbiased=True, 
    seed=914
):
    if not bootstrap:
        return accuracy_score_(y_true, y_pred)

    return bootstrap_score(
        y_true, 
        y_pred, 
        num_rounds=num_rounds, 
        ci=ci, 
        unbiased=unbiased, 
        func=accuracy_score_, 
        seed=seed
    )


def f1_score(
    y_true, 
    y_pred, 
    num_classes='auto', 
    average=None, 
    bootstrap=False, 
    num_rounds=200, 
    ci=.95, 
    unbiased=True, 
    seed=914
):
    if average is None:
        average = 'weighted'

    if not bootstrap:
        return f1_score_(y_true, y_pred, num_classes, average)

    return bootstrap_score(
        y_true, 
        y_pred, 
        num_rounds=num_rounds, 
        ci=ci, 
        unbiased=unbiased, 
        func=lambda t, p: f1_score_(t, p, average), 
        seed=seed
    )


def precision_score(
    y_true, 
    y_pred, 
    num_classes='auto', 
    average=None, 
    bootstrap=False, 
    num_rounds=200, 
    ci=.95, 
    unbiased=True, 
    seed=914
):
    """
    num_classes: 
        If 'auto', deduces the unique class labels from y_true
    """
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    if average not in [None, 'micro', 'macro', 'weighted']:
        raise ValueError('Wrong value of average parameter')
        
    if average is None:
        average = 'weighted'

    if num_classes == 'auto':
        num_classes = len(y_true.unique())
    func = Precision(average=average, num_classes=num_classes)

    if not bootstrap:
        return func(y_pred, y_true)

    return bootstrap_score(
        y_true, 
        y_pred, 
        num_rounds=num_rounds, 
        ci=ci, 
        unbiased=unbiased, 
        func=func, 
        seed=seed
    )


def recall_score(
    y_true, 
    y_pred, 
    num_classes='auto', 
    average=None, 
    bootstrap=False, 
    num_rounds=200, 
    ci=.95, 
    unbiased=True, 
    seed=914
):
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    if average not in [None, 'micro', 'macro', 'weighted']:
        raise ValueError('Wrong value of average parameter')

    if average is None:
        average = 'weighted'

    if num_classes == 'auto':
        num_classes = len(y_true.unique())
    num_classes = len(y_true.unique())
    func = Recall(average=average, num_classes=num_classes)

    if not bootstrap:
        return func(y_pred, y_true)

    return bootstrap_score(
        y_true, 
        y_pred, 
        num_rounds=num_rounds, 
        ci=ci, 
        unbiased=unbiased, 
        func=func, 
        seed=seed
    )


def bootstrap_score(
    y_true, 
    y_pred, 
    num_rounds, 
    ci=.95, 
    unbiased=True, 
    func=None, 
    seed=914
):
    bound = (1 - ci) / 2.
    check_output = func(y_true, y_pred)
    bootstrap_replicates = torch.zeros(num_rounds)
    for i in range(num_rounds):
        bootstrap_idx = torch.randint(len(y_true), (len(y_true),))
        y_true_bootstrap = torch.index_select(y_true, dim=0, index=bootstrap_idx)
        y_pred_bootstrap = torch.index_select(y_pred, dim=0, index=bootstrap_idx)
        score = func(y_true_bootstrap, y_pred_bootstrap)
        bootstrap_replicates[i] = score

    original = check_output
    standard_error = torch.std(bootstrap_replicates, unbiased=True)
    t = torch.sort(bootstrap_replicates, dim=0, descending=False)[0]
    upper_ci = torch.quantile(t, q=(ci + bound), dim=0)
    lower_ci = torch.quantile(t, q=bound, dim=0)
    return original, standard_error, (lower_ci, upper_ci)


def accuracy_score_(y_true, y_pred):
    return torch.mean(((y_true == y_pred).int()).float())


def f1_score_(y_true, y_pred, num_classes='auto', average=None):
    """
    References
    ----------
    1. https://stackoverflow.com/a/63358412
    """
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    if average not in [None, 'micro', 'macro', 'weighted']:
        raise ValueError('Wrong value of average parameter')
        
    def calc_f1_micro(y_true, y_pred):
        true_positive = torch.eq(y_true, y_pred).sum().float()
        f1_score = torch.div(true_positive, len(y_true))
        return f1_score

    def calc_f1_count_for_label(y_true, y_pred, label_id:int):
        # Compute label count
        true_count = torch.eq(y_true, label_id).sum()

        true_positive = torch.logical_and(
            torch.eq(y_true, y_pred),
            torch.eq(y_true, label_id)
        ).sum().float()
        precision = torch.div(
            true_positive, 
            torch.eq(y_pred, label_id).sum().float()
        )
        precision = torch.where(
            torch.isnan(precision),
            torch.zeros_like(precision).type_as(true_positive),
            precision
        )
        recall = torch.div(true_positive, true_count)
        f1 = 2 * precision * recall / (precision + recall)
        f1 = torch.where(
            torch.isnan(f1), 
            torch.zeros_like(f1).type_as(true_positive), 
            f1
        )
        return f1, true_count

    if average == 'micro':
        return calc_f1_micro(y_pred, y_true)

    if num_classes == 'auto':
        num_classes = len(y_true.unique())

    score = 0
    for label_id in range(0, num_classes + 1):
        f1, true_count = calc_f1_count_for_label(y_pred, y_true, label_id)

        if average == 'weighted':
            score += f1 * true_count
        elif average == 'macro':
            score += f1

    if average == 'weighted':
        score = torch.div(score, len(y_true))
    elif average == 'macro':
        score = torch.div(score, num_classes)

    return score


def test_f1_score():
    import numpy as np
    from tqdm.auto import tqdm
    from sklearn.metrics import f1_score as f1_score_sklearn

    f1_score_pytorch = f1_score
    errors = 0
    for _ in tqdm(range(10), desc='Testing F1 score'):
        y_true_pt = torch.randint(1, 10, (4096, 100)).flatten()
        y_pred_pt = torch.randint(1, 10, (4096, 100)).flatten()
        y_true_np = y_true_pt.numpy()
        y_pred_np = y_pred_pt.numpy()

        for av in ['micro', 'macro', 'weighted']:
            pt_pred = f1_score_pytorch(y_pred_pt, y_true_pt, average=av)
            sk_pred = f1_score_sklearn(y_true_np, y_pred_np, average=av)
            
            if not np.isclose(pt_pred, sk_pred):
                print('!' * 50)
                print(pt_pred, sk_pred, av)
                errors += 1

    if errors == 0:
        print('No errors!')


def test_accuracy_score():
    import numpy as np
    from tqdm.auto import tqdm
    from sklearn.metrics import accuracy_score as accuracy_score_sklearn

    accuracy_score_pytorch = accuracy_score
    errors = 0
    for _ in tqdm(range(10), desc='Testing accuracy score'):
        y_true_pt = torch.randint(1, 10, (4096, 100)).flatten()
        y_pred_th = torch.randint(1, 10, (4096, 100)).flatten()
        y_true_np = y_true_pt.numpy()
        y_pred_np = y_pred_th.numpy()

        pt_pred = accuracy_score_pytorch(y_pred_th, y_true_pt)
        sk_pred = accuracy_score_sklearn(y_true_np, y_pred_np)
        
        if not np.isclose(pt_pred, sk_pred):
            print('!' * 50)
            print(pt_pred, sk_pred)
            errors += 1

    if errors == 0:
        print('No errors!')


def test_precision_score():
    import numpy as np
    from tqdm.auto import tqdm
    from sklearn.metrics import precision_score as precision_score_sklearn

    precision_score_pytorch = precision_score
    errors = 0
    for _ in tqdm(range(10), desc='Testing precision score'):
        y_true_pt = torch.randint(10, (4096, 100)).flatten()
        y_pred_th = torch.randint(10, (4096, 100)).flatten()
        y_true_np = y_true_pt.numpy()
        y_pred_np = y_pred_th.numpy()

        for av in ['micro', 'macro', 'weighted']:
            pt_pred = precision_score_pytorch(y_pred_th, y_true_pt, average=av)
            sk_pred = precision_score_sklearn(y_true_np, y_pred_np, average=av)
            
            if not np.isclose(round(pt_pred.item()), round(sk_pred)):
                print('!' * 50)
                print(pt_pred, sk_pred, av)
                errors += 1

    if errors == 0:
        print('No errors!')


def test_recall_score():
    import numpy as np
    from tqdm.auto import tqdm
    from sklearn.metrics import recall_score as recall_score_sklearn

    recall_score_pytorch = recall_score
    errors = 0
    for _ in tqdm(range(10), desc='Testing recall score'):
        y_true_pt = torch.randint(10, (4096, 100)).flatten()
        y_pred_th = torch.randint(10, (4096, 100)).flatten()
        y_true_np = y_true_pt.numpy()
        y_pred_np = y_pred_th.numpy()

        for av in ['micro', 'macro', 'weighted']:
            pt_pred = recall_score_pytorch(y_pred_th, y_true_pt, average=av)
            sk_pred = recall_score_sklearn(y_true_np, y_pred_np, average=av)
            
            if not np.isclose(round(pt_pred.item()), round(sk_pred)):
                print('!' * 50)
                print(pt_pred, sk_pred, av)
                errors += 1

    if errors == 0:
        print('No errors!')