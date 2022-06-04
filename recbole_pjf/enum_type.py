# @Time   : 2022/3/11
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
recbole_pjf
"""

from enum import Enum


class FeatureSource(Enum):
    """Source of features.

    - ``INTERACTION``: Features from ``.inter`` (other than ``user_id`` and ``item_id``).
    - ``USER``: Features from ``.user`` (other than ``user_id``).
    - ``ITEM``: Features from ``.item`` (other than ``item_id``).
    - ``USER_SENTS``: Features from ``.udoc`` (other than ``user_id``).
    - ``ITEM_SENTS``: Features from ``.idoc`` (other than ``item_id``).
    - ``USER_ID``: ``user_id`` feature in ``inter_feat`` and ``user_feat``.
    - ``ITEM_ID``: ``item_id`` feature in ``inter_feat`` and ``item_feat``.
    - ``KG``: Features from ``.kg``.
    - ``NET``: Features from ``.net``.
    """

    INTERACTION = 'inter'
    USER = 'user'
    ITEM = 'item'
    USER_SENTS = 'udoc'
    ITEM_SENTS = 'idoc'
    USER_ID = 'user_id'
    ITEM_ID = 'item_id'
    KG = 'kg'
    NET = 'net'
