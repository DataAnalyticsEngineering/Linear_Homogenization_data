import torch

def get_data(data_loader, device='cpu', dtype=None):
    """[summary]

    :param data_loader: [description]
    :type data_loader: [type]
    :param device: [description], defaults to 'cpu'
    :type device: str, optional
    :return: [description]
    :rtype: [type]
    """
    if dtype is None:
        dtype = torch.float64

    x_list, y_list = [], []
    for x_batch, y_batch in list(data_loader):
        x_list.append(x_batch)
        y_list.append(y_batch)
    x = torch.cat(x_list)
    y = torch.cat(y_list)
    return x.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype)