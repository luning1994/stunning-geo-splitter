"""Return a list of countries from address data."""

from __future__ import print_function
import json
import logging
import os
import re
from time import sleep
import math
import shutil


import numpy as np
import pandas as pd
import requests
from openpyxl import load_workbook

mapping_counter = [0,0]
# init paths
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
exceed_google_flag = False
log_file = "address_parse.log"
logging.basicConfig(level=logging.INFO,filename=log_file,filemode='a',format='(%(asctime)s)(%(name)s:%(levelname)s):%(message)s')
##############################################################################
# Common geocoding functionalities (tfdl.utils.geocode)

LOG = logging.getLogger(__name__)

num_calls_to_google = 0

def query_country_google(address_str):
    """Query Google Maps Geocoding API to get country of address_str.

    Arguments:
        address_str: str - address string to be queried
    Return:
        country of the address
    """
    # if address_str is empty immediately return ''
    global num_calls_to_google
    global exceed_google_flag
    if address_str and not exceed_google_flag:
        url = 'https://maps.googleapis.com/maps/api/geocode/json?address={}'.format(
            address_str)
        attempt = 1
        while attempt <= 3:
            try:
                num_calls_to_google += 1
                return_object = requests.get(url).json()
                status = return_object['status']
                if status == "OVER_QUERY_LIMIT":
                    exceed_google_flag = True
                    LOG.error('OVER_QUERY_LIMIT received from Googled service. Subsequent calls disabled.')
                address_comps = return_object['results'][
                    0]['address_components']
                for comp in address_comps:
                    if 'country' in comp['types']:
                        return comp['long_name']
                break
            except (IndexError, KeyError):
                return ''
            except (requests.Timeout, requests.ConnectionError, KeyError) as e:
                if attempt == 3:
                    LOG.error('[Error] Unable to connect to Google service after three attempts. Subsequent calls disabled.')
                    exceed_google_flag = True
            except BaseException:  # all other exceptions, sleep and retry
                sleep(0.5)
            attempt += 1
    return ''

def update_country_result(address_str, cache=None, offline_mode=False, refresh_cache=False):
    """Subprocess to update a country result using cache.

    Arguments:
        address_str: str - address string to pass to
        cache: dict - result cache
        offline_mode: bool - whether to use cache-only mode
        refresh_cache: bool - whether to overwrite existing cache
    Return:
        country result
    """
    print("Requesting result for {}".format(address_str))
    cache_entry = re.sub(r'\W+', '', address_str).lower()
    if cache is not None:
        cached = cache_entry in cache.keys()
    else:
        cached = False
    if offline_mode or exceed_google_flag:
        if cached:
            tmp = cache[cache_entry]
            LOG.debug('result acquired from cache for %s: %s',
                      address_str, tmp)
            return tmp
        else:
            LOG.debug('no result acquired from cache for %s', address_str)
            return ''
    else:
        if cached and not refresh_cache:
            tmp = cache[cache_entry]
            LOG.debug('result acquired from cache for %s: %s',
                      address_str, tmp)
            return tmp
        else:

            tmp = ("result_addr_part","result_city","result_country")
            # tmp = query_country_google(address_str)
            cache[cache_entry] = tmp
            LOG.debug('result acquired for %s: %s, updating cache', address_str, tmp)
            return tmp


##############################################################################
# Use case specific code

INPUT_DIR = os.path.join(CUR_DIR, 'input')
if not os.path.exists(INPUT_DIR):
    os.makedirs(INPUT_DIR)
OUTPUT_DIR = os.path.join(CUR_DIR, 'output')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
CACHE_DIR = os.path.join(CUR_DIR, 'cache')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
LOOKUP_FILE = os.path.join(CACHE_DIR, 'lookup.json')
CACHE_FILE = os.path.join(CACHE_DIR, 'cache.json')

###################
# Utility functions


def clean_addr_part(str_):
    """Clean an address part.

    Arguments:
        str_: str - address part to clean
    Returns:
        cleaned address part
    """
    # remove all non-alpha-numeric characters
    str_ = re.sub(r'\W+', ' ', str_)
    # remove number sequences longer than 6 (not valid house no/ postal)
    str_ = re.sub(r'\d{7,}', ' ', str_)
    # remove extra spaces
    str_ = re.sub(r'\s+', ' ', str_)
    return str_.strip()


def check_addr_part(str_):
    """Check whether an address part is valid.

    Only valid address parts would be queried for country results.

    Arguments:
        str_: str - address part to check
    Returns:
        bool
    """
    str_ = str_.upper()
    if "ATTN" in str_:
        return False
    str_ = " "+ str_ + " "
    str_ = re.sub("\d", " ", str_)
    for w in ["FLOOR", "ROOM", "RM", "UNIT", "INC", "LLC", "LTD"]:
        str_ = re.sub(" {} ".format(w), " ", str_)
    str_ = re.sub("\s+", " ", str_)
    length = len(str_) >= 5
    if length:
        length_each_part = max([len(x) for x in str_.split()]) >= 4
        return length_each_part
    else:
        return False

###################
# Processing cache and lookup


def load_lookup_cache():
    """Load lookup and cache from disk.

    Arguments:
        none
    Returns:
        lookup and cache as dicts
    """
    # check whether lookup and cache files exist
    # if yes deserialize to dicts, if no return empty dicts
    if os.path.exists(LOOKUP_FILE):
        with open(LOOKUP_FILE, 'r') as json_:
            lookup = json.load(json_)
        LOG.info('lookup loaded')
    else:
        lookup = {}
        LOG.info('lookup initialized')
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as json_:
            cache = json.load(json_)
        LOG.info('cache loaded')
    else:
        cache = {'parses': {}, 'results': {}}
        LOG.info('cache initialized')
    return lookup, cache


def write_lookup_cache(lookup, cache, lookup_pretty=True, cache_pretty=False):
    """Write lookup and cache back to disk.

    Arguments:
        lookup: dict/None - lookup
        cache: dict/None - cache
        lookup_pretty: bool(True) - whether to pretty-print lookup
        cache_pretty: bool(False) - whether to pretty-print cache
    Returns:
        none
    """
    # pretty-print means indent=4
    if lookup is not None:
        with open(LOOKUP_FILE, 'w') as json_:
            if lookup_pretty:
                json.dump(lookup, json_, indent=4, sort_keys=True)
            else:
                json.dump(lookup, json_, sort_keys=True)
        LOG.info('lookup written')
    if cache is not None:
        with open(CACHE_FILE, 'w') as json_:
            if cache_pretty:
                json.dump(cache, json_, indent=4, sort_keys=True)
            else:
                json.dump(cache, json_, sort_keys=True)
        LOG.info('cache written')


def crunch_cache():
    """Crunch the cache file."""
    _, cache = load_lookup_cache()

    # crunched_parses = dict()
    crunched_results = dict()
    # parses_cache = cache['parses']
    results_cache = cache['results']

    # crunch caches
    # for key, value in parses_cache.items():
    #     cleaned = re.sub(r'\W+', '', key).lower()
    #     if cleaned not in crunched_parses.keys():
    #         crunched_parses[cleaned] = value
    for key, value in results_cache.items():
        cleaned = re.sub(r'\W+', '', key).lower()
        if cleaned not in crunched_results.keys():
            crunched_results[cleaned] = [value]
        else:
            crunched_results[cleaned].append(value)

    # clean up and rewrite
    for key in crunched_results:
        value = crunched_results[key]
        crunched_results[key] = max(set(value), key=value.count)
    # cache['parses'] = crunched_parses
    cache['results'] = crunched_results
    write_lookup_cache(lookup=None, cache=cache)
    LOG.info('crunch_cache finished')


def beautify_cache():
    """Beautify the cache file."""
    _, cache = load_lookup_cache()
    write_lookup_cache(lookup=None, cache=cache, cache_pretty=True)
    LOG.info('beautify_cache finished')

###################
# Batch-processing data


def process_naive(lookup, results_file, work_list=None):
    """Subprocess to do naive lookup of country names.

    Arguments:
        lookup: dict - lookup
        results_file: str - path to results file to write
        work_list: list/ None - list of keys in lookup to process
    Returns:
        path to results file
        items in work list with empty results
    """
    COUNTRY_WEIGHT = 1.0
    COUNTRY_ADDITIONAL_WEIGHT = 1.0

    CNTRY_LKUP = {}
    CNTRY_ADTNAL = {}

    country_list = [x.upper() for x in CNTRY_LKUP.values()]
    # country_code_list = [x.upper() for x in CNTRY_LKUP.keys()]
    country_additional_list = [x.upper() for x in CNTRY_ADTNAL.keys()]
    country_list_patterns = {key:re.compile('(\W+|^)' + key + "(\W+|$)") for key in country_list}
    country_additional_list_patterns = {key:re.compile('(\W+|^)' + key + "(\W+|$)") for key in country_additional_list}

    results = dict()

    # set work list to all items if it is None
    if work_list is None:
        work_list = lookup.keys()
    # 'real' work list (prevent errors when work_list items are not in lookup)
    work_list = set(lookup.keys()).intersection(set(work_list))

    work_list_remaining = list()
    for key in work_list:
        LOG.debug('processing key %s', key)
        value = lookup[key]
        # addr = value['addr_norm']
        if "key_pri" in value and value['key_pri'] != '':
            pri_entry = lookup[value['key_pri']]
            if pri_entry['country']:
                LOG.info('key_pri {} has been mapped to {} for key {}'.format(value['key_pri'], pri_entry['country'], key))
                continue

        results[key] = ""
        tmp = {}
        for i, addr in enumerate(value['addr_parts_norm']):
            index_weight = float(len(value['addr_parts_norm']) - i) / len(value['addr_parts_norm'])
            for country in country_list:
                p = country_list_patterns[country]
                search_result = re.search(p, addr)
                if search_result:
                    score = COUNTRY_WEIGHT * math.sqrt((len(country)+ 1)/( len(addr) + 1)) * index_weight
                    if country not in tmp:
                        tmp[country] = 0
                    tmp[country] += score
            for ca in country_additional_list:
                p = country_additional_list_patterns[ca]
                search_result = re.search(p, addr)
                if search_result:

                    score = COUNTRY_ADDITIONAL_WEIGHT * math.sqrt((len(ca)+ 1) / (len(addr) + 1)) * index_weight
                    if "BRANCH" in addr[search_result.span()[1]:].upper():
                        score += 0.5
                    country = CNTRY_ADTNAL[ca].upper()
                    if country not in tmp:
                        tmp[country] = 0
                    tmp[country] += score
        if len(tmp) == 1:
            only_country = list(tmp.keys())[0]
            if tmp[only_country] >0.1:
                results[key] = only_country
                LOG.info('result acquired: %s among %s %s', only_country, tmp, value['addr_norm'])
            else:
                results[key] = ''
                LOG.info('result forgone: %s among %s %s', only_country, tmp, value['addr_norm'])
                work_list_remaining.append(key)
        else:
            max_value = 0
            max_key = None
            for country in tmp:
                if tmp[country] > max_value:
                    max_key = country
                    max_value = tmp[country]
            for country in tmp:
                if country != max_key:
                    max_value -= tmp[country]
            if max_value > 0.6:
                results[key] = max_key
                LOG.info('result acquired: %s among %s %s', max_key, tmp, value['addr_norm'])
            else:
                if len(tmp) > 1:
                    LOG.info('result forgone: %s among %s %s', max_key, tmp, value['addr_norm'])
                results[key] = ''
                work_list_remaining.append(key)

    # output
    with open(results_file, 'w') as json_:
        json.dump(results, json_, indent=4, sort_keys=True)
    LOG.info('process_naive finished, %s/%s remaining',len(work_list_remaining), len(work_list))
    return results_file, work_list_remaining


def process_google(lookup, cache):
    for key in lookup:
        LOG.debug('processing key %s', key)
        value = lookup[key]
        if value["addr_sub"] and value["city"] and value["country"]:
            continue
        tmp = update_country_result(value['addr_norm'], cache)
        if tmp is not None:
            lookup[key]['addr_sub'] = tmp[0].upper()
            lookup[key]['city'] = tmp[1].upper()
            lookup[key]['country'] = tmp[2].upper()


###################
# Main processes

def normalize_addr(addr_raw):
    # TODO: Implement this function
    addr = addr_raw.strip()
    return addr


def main(file_in):
    """Process the file.

    Arguments:
        config: dict - configurations
    """
    # config
    cache = {}
    keys = []
    lookup = {}
    file_out = "result.xlsx"
    shutil.copy2(file_in, file_out)
    df_ = pd.read_excel(file_in).fillna('')
    for _, row in df_.iterrows():
        row_list = [row["Line 1"], row["Line 2"], row["Line 3"], row["Line 4"]]
        row_str = " ".join(row_list)
        splitted = row_str.split("ADD.",1)
        key = None
        addr_raw = splitted[1] if len(splitted)==2 else None
        if addr_raw is not None and check_addr_part(addr_raw.upper()):
            addr_key = re.sub("[^a-zA-Z0-9]","", addr_raw).upper()
            if addr_key:
                key = addr_key
                lookup[addr_key] = {
                    "addr_norm": normalize_addr(addr_raw),
                    "addr_sub": None,
                    "city": None,
                    "country": None
                }
        keys.append(key)

    df_["key"] = pd.Series(keys)
    # path = os.path.join(OUTPUT_DIR, 'results_naive.json')
    # process_naive(lookup, path)
    process_google(lookup, cache)
    addr_sub_list = []
    city_list = []
    country_list = []

    for _, row in df_.iterrows():
        addr_sub = city = country = None
        if row["key"] and row["key"] in lookup:
            value = lookup[row["key"]]
            addr_sub = value["addr_sub"]
            city = value["city"]
            country = value["country"]
        addr_sub_list.append(addr_sub)
        city_list.append(city)
        country_list.append(country)
    df_["Address"] = pd.Series(addr_sub_list)
    df_["City"] = pd.Series(city_list)
    df_["Country"] = pd.Series(country_list)

    book = load_workbook(file_out)
    writer = pd.ExcelWriter(file_out, engine='openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df_.to_excel(writer, index=False)
    writer.save()
    LOG.info('written output to %s', file_out)

    LOG.info('process finished for %s', file_in)


if __name__ == '__main__':
    # parse arguments
    main("Remitter and Bene information.xlsx")
