import math


def cal_ncoef(budget, im_size, channel, nclass, nbasis, ds=2, return_total=False):
    ncoef = (budget * (im_size * im_size * channel * nclass) - nbasis * im_size * im_size * channel / (ds * ds)) / (nclass * nbasis)
    total_alloc = nbasis * im_size * im_size * channel / (ds * ds) + math.floor(ncoef) * nbasis * nclass
    total_budget = budget * (im_size * im_size * channel * nclass)
    print('total  alloc:', total_alloc)
    print('budget total:', total_budget)
    if return_total:
        return ncoef, total_alloc, total_budget
    else:
        return ncoef
