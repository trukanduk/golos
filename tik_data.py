# -*- coding: utf8 -*-

from datetime import datetime
import itertools
import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    'Candidate',
    'parse_candidate_by_text',
    'TikData',
    'RootHumanTikData',
    'MunicipalVotes',
    'build_root_tik',
    'load_root_tik_data_from_dict'
]

class Candidate(object):
    def __init__(self, id, title, alternative_titles=[], is_human=False):
        self.id = id
        self.title = title
        self.alternative_titles = list(alternative_titles)
        self.is_human = is_human

    @classmethod
    def from_dict(cls, d):
        return cls(d['id'], d['title'], is_human=d['is_human'])

    def is_this(self, text):
        lower_text = text.lower()
        for title in [self.title] + self.alternative_titles:
            if title.lower() in lower_text:
                return True
        return False

    def to_dict(self):
        return {'id': self.id, 'title': self.title, 'is_human': self.is_human}

    def __eq__(self, other):
        return self.id == other.id

    def make_against_candidate(self):
        return Candidate(self.id + '_Against', self.title + u' - Против', is_human=self.is_human)


AVAILABLE_PARTIES = [
    Candidate('SpravRos', u'Справедливая Россия'),
    Candidate('ER', u'Единая Россия'),
    Candidate('KPRF', u'КПРФ', [u'Коммунистическая партия российской федерации']),
    Candidate('KomRos', u'Коммунисты России'),
    Candidate('Pens', u'Партия пенсионеров'),
    Candidate('Rodina', u'Родина'),
    Candidate('LDPR', u'ЛДПР', [u'Либерально-демократическая партия россии']),
    Candidate('Yabloko', u'Яблоко'),
    Candidate('Zelenye', u'Зелёные', [u'Зеленые']),
    Candidate('Kazaki', u'Казачья'),
    Candidate('Rosta', u'Партия роста'),
    Candidate('Patrioty', u'Патриоты России'),
]


# FIXME:
# Лейте Перейра де Сена Татьяна Евгеньевна
HUMAN_CANDIDATE_RE = re.compile(u'^([А-Яа-яЁё][a-яё-]+ ){,5}[А-Яа-яЁё][a-яё-]+$', re.UNICODE)


def parse_candidate_by_text(candidate_text):
    candidate_text = candidate_text.strip()
    if HUMAN_CANDIDATE_RE.match(candidate_text) is not None and u'партия' not in candidate_text.lower():
        logger.debug('Assume candidate with text "%s" is a human', candidate_text)
        return Candidate(candidate_text.replace(' ', '_'), candidate_text, is_human=True)

    for party in AVAILABLE_PARTIES:
        if party.is_this(candidate_text):
            logger.debug('Candidate text "%s" is %s', candidate_text, party.id)
            return party

    raise ValueError(u'Unknown candidate by text: "{}"'.format(candidate_text))


class TikChildInfo(object):
    def __init__(self, id, url, uiks):
        self.id = id
        self.url = url
        self.uiks = uiks

    @classmethod
    def from_dict(cls, d):
        return cls(d['id'], d['url'], d['uiks'])

    def to_dict(self):
        return {
            'id': self.id,
            'url': self.url,
            'uiks': self.uiks,
        }

    def __repr__(self):
        return '"{}": {}'.format(self.id, self.uiks)


class TikData(object):
    STATIC_ROWS = [
        'total_izb',
        'ballots_recieved',
        'ballots_dosrok',
        'ballots_indoor',
        'ballots_outdoor',
        'ballots_unused',
        'ballots_mobile',
        'ballots_stationar',
        'ballots_not_valid',
        'ballots_valid',
        'ballots_lost',
        'ballots_extra',
    ]

    def __init__(self, id, candidates, uiks, url, data, children=[]):
        self.id = id
        self.candidates = candidates
        self.uiks = uiks
        candidate_rows = list(itertools.chain.from_iterable([(candidate.id, candidate.id + '_share') for candidate in self.candidates]))
        self.url = url
        self.data = pd.DataFrame(data, columns=TikData.STATIC_ROWS + candidate_rows, index=self.uiks)
        self.children_info = list(children)

    @classmethod
    def from_raw_data(cls, id, candidates, uiks, url, raw_data):
        raw_data = np.array(raw_data, dtype=np.float32)
        static_data = raw_data[: len(TikData.STATIC_ROWS)]
        candidates_data = raw_data[len(TikData.STATIC_ROWS) :]
        assert candidates_data.shape[0] == len(candidates)

        num_ballots = static_data[TikData.STATIC_ROWS.index('ballots_valid')] \
            + static_data[TikData.STATIC_ROWS.index('ballots_not_valid')]
        extended_candidates_data = np.zeros((len(candidates) * 2, len(uiks)), dtype=np.float32)
        for candidate_index, candidate_data in enumerate(candidates_data):
            extended_candidates_data[candidate_index * 2] = candidate_data
            extended_candidates_data[candidate_index * 2 + 1] = candidate_data / num_ballots
        data = np.concatenate((static_data, extended_candidates_data)).transpose()
        return cls(id, candidates, uiks, url, data)

    @classmethod
    def from_children(cls, id, url, children):
        candidates = children[0].candidates
        children_info = []
        uiks = []
        data = []
        for child in children:
            assert not child.children_info
            assert not child.is_human_candidates()
            assert child.candidates == candidates
            children_info.append(TikChildInfo(child.id, child.url, child.uiks))
            uiks += child.uiks
            data.append(child.data)
        data = pd.concat([child.data for child in children])
        assert len(data) == len(uiks)
        return TikData(id, candidates, uiks, url, data, children_info)


    @classmethod
    def from_dict(cls, d):
        candidates = [Candidate.from_dict(candidate) for candidate in d['candidates']]
        children_info = [TikChildInfo.from_dict(oik) for oik in d['children_info']]
        return cls(d['id'], candidates, d['uiks'], d['url'], d['data'], children_info)

    def is_human_candidates(self):
        return all(map(lambda candidate: candidate.is_human, self.candidates))

    def to_dict(self):
        return {
            'id': self.id,
            'candidates': [candidate.to_dict() for candidate in self.candidates],
            'uiks': self.uiks,
            'data': self.data.values.tolist(),
            'url': self.url,
            'children_info': [oik.to_dict() for oik in self.children_info],
        }


class RootHumanTikData(object):
    def __init__(self, id, url, children):
        self.id = id
        self.url = url
        self.children = children

    @classmethod
    def from_dict(cls, d):
        return cls(d['id'], d['url'], [TikData.from_dict(child) for child in d['children']])

    def to_dict(self):
        return {
            'id': self.id,
            'url': self.url,
            'children': [child.to_dict() for child in self.children],
        }


def load_root_tik_data_from_dict(d):
    if 'data' in d:
        return TikData.from_dict(d)
    else:
        return RootHumanTikData.from_dict(d)


class MunicipalVotes(object):
    DATE_FORMAT = '%Y-%m-%d'

    def __init__(self, id, town, tik_id, date, url, votes_data=[]):
        self.id = id
        self.tik_id = tik_id
        self.town = town or tik_id
        self.date = date
        self.url = url
        self.votes_data = list(votes_data)

    @classmethod
    def from_dict(cls, d):
        votes_date = datetime.strptime(d['date'], cls.DATE_FORMAT).date()
        votes_data = [load_root_tik_data_from_dict(datum) for datum in d['votes_data']]
        return cls(d['id'], d['town'], d['tik_id'], votes_date, d['url'], votes_data)

    def to_dict(self):
        return {
            'id': self.id,
            'town': self.town,
            'tik_id': self.tik_id,
            'date': self.date.strftime(self.DATE_FORMAT),
            'url': self.url,
            'votes_data': [datum.to_dict() for datum in self.votes_data]
        }

    def add_datum(self, votes_datum):
        self.votes_data.append(votes_datum)


def build_root_tik(tik_id, url, children):
    all_human = all(map(lambda child: child.is_human_candidates(), children))
    no_human = all(map(lambda child: not child.is_human_candidates(), children))
    assert all_human != no_human

    if all_human:
        logger.debug('Create human root tik for %s', tik_id)
        return RootHumanTikData(tik_id, url, children)
    else:
        logger.debug('Create party root tik for %s', tik_id)
        return TikData.from_children(tik_id, url, children)

