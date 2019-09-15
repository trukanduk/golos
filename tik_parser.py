#!/usr/bin/env python3
# -*- coding: utf8 -*-

from collections import namedtuple
from datetime import datetime
from pprint import pprint
import itertools
import json
import logging
import os
import re
import sys

import bs4
import numpy as np
import pandas as pd
import requests
import click

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tik_data import *

if sys.version_info > (3, 0):
    unicode = str

CACHE_PAGE = True
CACHE_PATH = os.path.expanduser('~/.uiks/')
if CACHE_PAGE:
    import hashlib


logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def _init_cache_dir():
    if not os.path.isdir(CACHE_PATH):
        os.mkdir(CACHE_PATH)


def _make_cache_filepath(url):
    url_hash = hashlib.md5(url.encode('utf8')).hexdigest()
    cache_filename = 'page-cache-{}.html'.format(url_hash)
    return os.path.join(CACHE_PATH, cache_filename)


def _request_page_internal(url, params):
    if params is None:
        return requests.get(url).text
    else:
        return requests.post(url, params).text

def get_page_content(url, params=None, cache=True):
    if cache and CACHE_PAGE:
        _init_cache_dir()
        cache_filepath = _make_cache_filepath(url)
        if os.path.isfile(cache_filepath):
            with open(cache_filepath, 'r') as f:
                logger.debug('Use cache for html page %s: %s', url, cache_filepath)
                return f.read()

    result = _request_page_internal(url, params)

    if cache and CACHE_PAGE:
        with open(cache_filepath, 'w') as f:
            f.write(result)
    return result


def get_parsed_page(url, params=None, cache=True):
    content = get_page_content(url, params=params, cache=cache)
    return bs4.BeautifulSoup(content, 'lxml')


def _find_first_parent_tag(tag, tag_name):
    for parent_tag in tag.parents:
        if parent_tag is not None and parent_tag.name == tag_name:
            return parent_tag

def _find_next_sibling_with_tag(tag, tag_name):
    next_tag = tag.next_sibling
    while next_tag is not None:
        if not isinstance(next_tag, str) and not isinstance(next_tag, unicode) and next_tag.name == tag_name:
            return next_tag
        next_tag = next_tag.next_sibling
    return None


class TikResultsPageParser(object):
    logger = logging.getLogger('TikResultsPageParser')

    UikTables = namedtuple('UikTables', ['wrapper_table', 'header_table', 'results_table'])
    TIK_TITLE_MARKER = re.compile(u'Наименование( избирательной)? комиссии', re.UNICODE)
    UIK_HEADER_CELL_RE = re.compile(u'^\\s*УИК +№(\\d+)\\s*$')

    @classmethod
    def parse(cls, url, tik_id=None):
        soup = get_parsed_page(url)
        return cls.parse_soup(soup, url, tik_id)

    @classmethod
    def parse_soup(cls, soup, url, tik_id=None):
        cls.logger.debug('Processing page %s', url)
        if tik_id is None:
            cls.logger.debug('TIK id is not known, try to parse it')
            tik_id = cls.parse_tik_title(soup)
        tables = cls.find_uik_tables(soup)
        result = cls.parse_uiks_tables(tik_id, url, tables)
        cls.logger.debug('Done processing page %s', url)
        return result

    @classmethod
    def parse_tik_title(cls, soup):
        marker_tag = soup.find(text=cls.TIK_TITLE_MARKER)
        tr = _find_first_parent_tag(marker_tag, u'tr')
        result = tr.find_all(u'td')[1].string
        if result is None:
            cls.logger.warning('Cannot parse tik id')
        return result

    @classmethod
    def find_uik_tables(cls, soup):
        results_table = soup.find(cls.is_uik_results_table)
        if results_table is None:
            cls.logger.warning('Cannot find results table!')
            return None
        wrapper_table = _find_first_parent_tag(results_table, u'table')
        all_tables = wrapper_table.find_all(u'table')
        assert len(all_tables) == 2
        header_table = all_tables[0]
        return cls.UikTables(wrapper_table, header_table, results_table)

    @classmethod
    def parse_uiks_tables(cls, tik_id, url, tables):
        candidates, candidates_offset = cls.parse_candidates_from_header_table(tables.header_table)
        uiks, data = cls.parse_uiks_results_table(tables.results_table, candidates_offset)
        return TikData.from_raw_data(tik_id, candidates, uiks, url, data)

    @classmethod
    def parse_candidates_from_header_table(cls, table):
        candidates_trs = list(table.find_all(u'tr')[len(TikData.STATIC_ROWS) + 2 :])
        if not candidates_trs[0].find_all('td')[0].nobr.string:
            logger.debug('There is only one candidate?')
            assert len(candidates_trs) == 3
            assert cls.extract_candidate_text_from_row(candidates_trs[1]).lower() == u'за'
            assert cls.extract_candidate_text_from_row(candidates_trs[2]).lower() == u'против'
            candidate = cls.parse_candidate_from_row(candidates_trs[0])
            candidate_against = candidate.make_against_candidate()
            return [candidate, candidate_against], 1

        candidates = []
        for tr in candidates_trs:
            candidates.append(cls.parse_candidate_from_row(tr))
        return candidates, 0

    @classmethod
    def parse_candidate_from_row(cls, tr):
        candidate_text = cls.extract_candidate_text_from_row(tr)
        return parse_candidate_by_text(candidate_text)

    @classmethod
    def extract_candidate_text_from_row(cls, tr):
        return unicode(tr.find_all(u'td')[1].nobr.string).strip()

    @classmethod
    def is_uik_results_table(cls, table):
        if table.name != 'table':
            return False

        for td in table.find_all('tr')[0].find_all('td'):
            if td.name != 'td':
                return False

            if cls._parse_uik_number(td) is None:
                return False

        return True

    @classmethod
    def parse_uiks_results_table(cls, table, candidates_offset):
        data = []
        trs = list(table.find_all('tr'))
        uiks = cls.parse_uik_row(trs[0], cls._parse_uik_number)
        for tr in trs[1: len(TikData.STATIC_ROWS) + 1]:
            data.append(cls.parse_uik_row(tr, cls._parse_uik_results))

        for tr in trs[len(TikData.STATIC_ROWS) + 2 + candidates_offset:]:
            data.append(cls.parse_uik_row(tr, cls._parse_uik_results))

        return uiks, data

    @classmethod
    def _parse_uik_number(cls, td):
        match = cls.UIK_HEADER_CELL_RE.match(unicode(td.string))
        return match and match.group(1)

    @classmethod
    def _parse_uik_results(cls, td):
        return float(td.find(u'b').string.strip())

    @classmethod
    def parse_uik_row(cls, tr, parser):
        result = []
        for td in tr.find_all('td'):
            result.append(parser(td))
        return result


class RootTikPageParser(object):
    logger = logging.getLogger('RootTikPageParser')

    ChildTik = namedtuple('ChildTik', ['title', 'url'])

    @classmethod
    def parse(cls, url):
        cls.logger.debug('Going to parse root tik page: %s', url)
        soup = get_parsed_page(url)
        child_tiks = cls.collect_child_tiks(soup)
        if child_tiks is None:
            cls.logger.info('Root tik page %s seems to be non-hierarchical', url)
            return TikResultsPageParser.parse_soup(soup, url)

        result = cls.parse_by_child_tiks(soup, url, child_tiks)
        cls.logger.debug('Done root tik %s', url)
        return result

    @classmethod
    def parse_by_child_tiks(cls, soup, url, child_tiks):
        tik_id = TikResultsPageParser.parse_tik_title(soup)
        datas = []
        for child_tik in child_tiks:
            datas.append(TikResultsPageParser.parse(child_tik.url, tik_id=child_tik.title))
        return build_root_tik(tik_id, url, datas)

    @classmethod
    def collect_child_tiks(cls, soup):
        form = soup.find(u'form', attrs={'name': u'go_reg'})
        if form is None:
            return None
        select = form.find(u'select', attrs={'name': u'gs'})
        return [cls.ChildTik(option.string, option['value']) for option in select.find_all('option')[1:]]

class TikVotesPageParser(object):
    logger = logging.getLogger('TikVotesPageParser')

    ResultsPageInfo = namedtuple('ResultsPageInfo', ['title', 'url'])
    RESULT_TITLE = re.compile(u'Сводная таблица .*(результатов|итогов)')

    @classmethod
    def parse(cls, url, town=None):
        cls.logger.debug('Going to process page %s', url)
        votes, children = cls.parse_meta(url, town)
        for child in children:
            votes.add_datum(RootTikPageParser.parse(child.url))
        cls.logger.debug('Done processing page %s', url)
        return votes

    @classmethod
    def parse_meta(cls, url, town):
        soup = get_parsed_page(url)
        votes_title = cls.parse_votes_title(soup)
        tik_title = cls.parse_tik_title(soup)
        votes_date = cls.parse_votes_date(soup)
        result_pages = cls.parse_result_urls(soup)
        return MunicipalVotes(votes_title, town, tik_title, votes_date, url), result_pages

    @classmethod
    def parse_votes_title(cls, soup):
        marker_cell = soup.find_all(text=u'Сведения о выборах')
        assert len(marker_cell) == 1
        table_tag = _find_first_parent_tag(marker_cell[0], u'table')
        return unicode(table_tag.find_all('tr')[1].td.b.string)

    @classmethod
    def parse_tik_title(cls, soup):
        marker_cell = soup.find_all(text=u'Наименование комиссии')
        assert len(marker_cell) == 1
        row_tag = _find_first_parent_tag(marker_cell[0], u'tr')
        return unicode(row_tag.find_all('td')[1].string)

    @classmethod
    def parse_votes_date(cls, soup):
        marker_cell = soup.find_all(text=u'Дата голосования')
        assert len(marker_cell) == 1
        row_tag = _find_first_parent_tag(marker_cell[0], u'tr')
        return datetime.strptime(unicode(row_tag.find_all('td')[1].string), '%d.%m.%Y').date()

    @classmethod
    def parse_result_urls(cls, soup):
        marker_cell = soup.find_all(text=u'РЕЗУЛЬТАТЫ ВЫБОРОВ')
        assert len(marker_cell) == 1
        marker_row = _find_first_parent_tag(marker_cell[0], u'tr')
        current_row = marker_row.next_sibling
        result = []
        while current_row is not None:
            result_url = cls.parse_result_urls_row(current_row)
            if result_url:
                result.append(result_url)
            current_row = current_row.next_sibling
        assert result
        return result

    @classmethod
    def parse_result_urls_row(cls, row):
        if isinstance(row, str) or isinstance(row, unicode):
            return None
        if row['class'] != ['trReport']:
            cls.logger.warning('Unexpected non-report row: %s', row)
            return None

        link = row.find('a')
        title = unicode(link.string)
        is_report_row = cls.is_results_title(title)
        cls.logger.debug('Title "%s" seems %s to be result page', title, '' if is_report_row else 'NOT')
        if not is_report_row:
            return None

        return TikVotesPageParser.ResultsPageInfo(title, link['href'])

    @staticmethod
    def is_results_title(title):
        return TikVotesPageParser.RESULT_TITLE.search(title) is not None


class VotesListParser(object):
    logger = logging.getLogger('VotesListParser')
    URL = 'http://www.moscow_reg.vybory.izbirkom.ru/region/moscow_reg'

    VoteInfo = namedtuple('VoteInfo', ['town', 'url'])
    NUM_RESULTS_RE = re.compile(u'Всего найдено записей: \\d+')

    @classmethod
    def parse(cls, params=None):
        cls.logger.debug('Going to process page with params %s', params)
        vote_infos = cls.parse_meta(params)
        result = [TikVotesPageParser.parse(vote_info.url, town=vote_info.town) for vote_info in vote_infos]
        cls.logger.debug('Successfully processed page %s', params)
        return result

    @classmethod
    def parse_meta(cls, params):
        soup = get_parsed_page(cls.URL, params=params, cache=False)
        marker_cell = soup.find_all(text=cls.NUM_RESULTS_RE)
        assert len(marker_cell) == 1
        num_results_table = _find_first_parent_tag(marker_cell[0], u'table')
        main_table = _find_next_sibling_with_tag(num_results_table, u'table')

        rows = []
        last_town_title = None
        for tr in main_table.find_all('tr'):
            if tr.attrs.get('bgcolor') == '#555555':
                last_town_title = None
                continue

            last_town_title = cls.parse_town_title(tr.find_all('td')[0]) or last_town_title
            # cls.logger.debug('town title: %s', last_town_title)
            link = tr.find(u'a')
            if link is None or link['class'] != ['vibLink']:
                if link is None or link['class'] != ['voidVibLink']:
                    cls.logger.warning('No votes link: %s (%s)', tr, link)
                continue
            url = unicode(link['href'])
            rows.append(cls.VoteInfo(last_town_title, url))
        return rows

    @classmethod
    def parse_town_title(cls, td):
        for child in reversed(list(td.children)):
            if isinstance(child, str) or isinstance(child, unicode):
                child = child.strip()
                if child:
                    return child
        tag_b = td.find('b')
        if tag_b is not None:
            return unicode(tag_b.string)

        return None


@click.command()
@click.argument('out_dir')
@click.option('--dry/--no-dry', default=False)
def main(out_dir, dry):
    votes = VotesListParser.parse() # {'start_date': '01.03.2019', 'urovproved': 'all', 'vidvibref': 'all', 'vibtype': 'all', 'end_date': '31.01.2020', 'sxemavib': 'all', 'action': 'search_by_calendar', 'region': '50', 'ok': '%C8%F1%EA%E0%F2%FC'})
    if dry:
        logger.info('Dry run, skip saving')
        return
    for vote in votes:
        filename = '{}_{}.json'.format(vote.id.replace(u' ', u'_').replace(u'№', u'N'), vote.date.strftime('%Y-%m-%d'))
        with open(os.path.join(out_dir, filename), 'w') as f:
            json.dump(vote.to_dict(), f)


if __name__ == '__main__':
    main()
