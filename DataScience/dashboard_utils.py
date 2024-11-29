import pandas,ds_parse,json,collections,os,gzip,sys
import numpy as np
import argparse
import time
from DashboardMpi.helpers import command

def get_ts_5min_bin(ts):
    str_5min = ts[:14]
    x = int(float(ts[14:16])/5)*5
    if x < 10:
        str_5min += '0'
    str_5min += str(x)+':00Z'
    return str_5min

def get_prediction_prob(a0, pred_line):
    # parse probability of predicted action
    # this function assume that a0 is 0-index

    if ':' in pred_line:                           # prediction file has pdf of all actions (as in --cb_explore_adf -p)
        if ',' in pred_line:
            if pred_line.startswith(str(a0)+':'):
                sep = ':'
            else:
                sep = ','+str(a0)+':'
            pred_prob = float(ds_parse.extract_field(pred_line,sep,','))
        else:
            if pred_line.startswith(str(a0)+':'):
                pred_prob = 1
            else:
                print('Error: Prediction action (0) does not match log file action ({}) - log: {} - pred: {}'.format(a0,pred_line))
                sys.exit()
    else:                                          # prediction file has only one action (as in --cb_adf -p)
        pred_prob = 1 if a0 == int(pred_line) else 0

    return pred_prob


def output_dashboard_data(d, dashboard_file, commands={}, sep=':'):
    data_dict = collections.OrderedDict()
    for x in d:
        for type in d[x]:
            for field in d[x][type]:
                data_dict.setdefault(type+sep+field, []).append(d[x][type][field])

    df = pandas.DataFrame(data_dict, index=pandas.to_datetime([x for x in d]), dtype=float)

    df_col = collections.OrderedDict()
    for x in df.columns:
        temp = x.split(sep)
        df_col.setdefault(temp[0],[]).append(temp[1])

    agg_windows = [('5T',5),('10T',10),('30T',30),('H',60),('2H',2*60),('3H',3*60),('6H',6*60),('D',24*60),('7D',7*1440)]
    with open(dashboard_file, 'a') as f:
        for ag in agg_windows:
            for index, row in df.resample(ag[0]).agg({type+sep+field : max if field == 'c' else sum for type in df_col for field in df_col[type]}).replace(np.nan, 0.0).iterrows():
                d = []
                for type in df_col:
                    temp = collections.OrderedDict({field : row[type+sep+field] for field in df_col[type]})
                    temp["w"] = ag[1]
                    temp["t"] = type
                    d.append(temp)
                f.write(json.dumps({"ts":index.strftime("%Y-%m-%dT%H:%M:%SZ"),"d":d})+'\n')

        # total aggregates
        tot = df.agg({type+sep+field : max if field == 'c' else sum for type in df_col for field in df_col[type]}).replace(np.nan, 0.0)
        d = []
        for type in df_col:
            temp = collections.OrderedDict({field : tot[type+sep+field] for field in df_col[type]})
            temp["w"] = "tot"
            temp["t"] = type
            if type in commands.keys():
                temp["command"] = command.to_commandline(commands[type])
            d.append(temp)
        f.write(json.dumps({"ts":"Total","d":d})+'\n')
    print('Dashboard file:',dashboard_file)

def merge_and_unique_stats(stats_files, dashboard_file):
    d = {}
    for stats_file in ds_parse.input_files_to_fp_list(stats_files):
        print('Processing: {}'.format(stats_file))
        for x in open(stats_file, 'rb'):
            js = json.loads(x)
            if js['d'][0]['w'] != 5:
                continue
            num = None
            den = None
            for y in js['d']:
                if y['t'] == 'online':
                    num = y['n']
                    den = y['d']
                    break
            if num is None or den is None:
                print('Error: "online" policy stats not found in input:',x)
                continue
            if js['ts'] not in d or d[js['ts']]['online']['d'] < den or (d[js['ts']]['online']['d'] == den and d[js['ts']]['online']['n'] < num):
                d[js['ts']] = {y['t'] : {field : y[field] for field in y if field not in {'w','t'}} for y in js['d']}

    print('Output dashboard data...')
    output_dashboard_data(d, dashboard_file)

def create_stats(log_fp, log_type='cb', d=None, predictions_files=None, is_summary=False, report_progress=True):

    t0 = time.time()
    if d is None:
        d = {}

    if predictions_files is None:
        print('Searching prediction files for log file: {}'.format(log_fp))
        predictions_files = []
        for fn in os.scandir(os.path.dirname(log_fp)):
            if fn.path.startswith(log_fp+'.') and fn.name.endswith('.pred'):
                predictions_files.append(fn.path)

    # load predictions from predictions_files
    pred = {}
    for pred_fp in predictions_files:
        if os.path.isfile(pred_fp):
            if is_summary:
                name = pred_fp.split('/')[-1].split('.')[-2]
            else:
                name = pred_fp.split('.')[-2] # check that policy name is encoded in file_name
            if name:
                if log_type == 'cb':
                    pred[name] = [x.strip() for x in open(pred_fp) if x.strip()]
                elif log_type == 'ccb':
                    with open(pred_fp) as f:
                        pred[name] = []
                        slot = []
                        for x in f:
                            x = x.strip()
                            if x:
                                slot.append(x)
                            else:
                                pred[name].append(slot)
                                slot = []
                print('Loaded {} predictions from {}'.format(len(pred[name]), pred_fp))
            else:
                print('Name is not valid - Skip: {}'.format(pred_fp))
        else:
            print('Error loading policy predictions. Pred file not found: {}'.format(pred_fp))
            sys.exit()

    if len(pred) > 1 and min(len(pred[name]) for name in pred) != max(len(pred[name]) for name in pred):
        print('Error: Prediction file length ({}) must be equal for all files'.format([len(pred[name]) for name in pred]))
        sys.exit()

    print('Processing: {}'.format(log_fp))
    bytes_count = 0
    tot_bytes = os.path.getsize(log_fp)
    evts = 0
    for i,x in enumerate(gzip.open(log_fp, 'rb') if log_fp.endswith('.gz') else open(log_fp, 'rb')):
        if report_progress:
            # display progress
            bytes_count += len(x)
            if (i+1) % 1000 == 0:
                if log_fp.endswith('.gz'):
                    ds_parse.update_progress(i+1)
                else:
                    ds_parse.update_progress(bytes_count,tot_bytes)

        data = None

        if log_type == 'ccb':
            if x.startswith(b'{"Timestamp"') and x.strip().endswith(b'}'):
                data = ds_parse.ccb_json_cooked(x)
                aggregates_ccb_data(data, pred, d, evts)

        elif log_type == 'cb':
            if is_summary:
                data = json.loads(x.decode("utf-8"))
            elif x.startswith(b'{"_label_cost":') and x.strip().endswith(b'}'):
                data = ds_parse.json_cooked(x, do_decode=True)

            # Skip wrongly formated lines or not activated lines
            if data is None or data['skipLearn']:
                continue

            aggregates_cb_data(data, pred, d, evts)
        evts += 1

    if report_progress:
        if log_fp.endswith('.gz'):
            len_text = ds_parse.update_progress(i+1)
        else:
            len_text = ds_parse.update_progress(bytes_count,tot_bytes)
        sys.stdout.write("\r" + " "*len_text + "\r")
        sys.stdout.flush()

    print('Read {} lines - Processed {} events'.format(i+1, evts))

    if any(len(pred[name]) != evts for name in pred):
        print('Error: Prediction file length ({}) is different from number of events in log file ({})'.format([len(pred[name]) for name in pred], evts))
        sys.exit()
    print('Total Elapsed Time: {:.1f} sec.'.format(time.time()-t0))
    return d


def aggregates_cb_data(data, pred, d, evts):
    if data['cost'] == b'0':
        r = 0
        abs_r = 0
    else:
        r = -float(data['cost'])
        abs_r = abs(r)

    ############################### Aggregates for each bin ######################################
    #
    # 'n':   IPS of numerator
    # 'N':   total number of samples in bin from log (IPS = n/N)
    # 'd':   IPS of denominator (SNIPS = n/d)
    # 'Ne':  number of samples in bin when off-policy agrees with log policy
    # 'c':   max abs. value of numerator's items (needed for Clopper-Pearson confidence intervals)
    # 'SoS': sum of squares of numerator's items (needed for Gaussian confidence intervals)
    #
    #################################################################################################

    # binning timestamp every 5 min
    ts_bin = get_ts_5min_bin(data['ts'])

    # initialize aggregates for ts_bin
    if ts_bin not in d:
        d[ts_bin] = collections.OrderedDict([
            ('online', {'n': 0, 'N': 0, 'd': 0}),
            ('baseline1', {'n': 0., 'N': 0, 'd': 0., 'Ne': 0, 'c': 0., 'SoS': 0}),
            ('baselineRand', {'n': 0., 'N': 0, 'd': 0., 'Ne': 0, 'c': 0., 'SoS': 0})
        ])
        for name in pred:
            d[ts_bin][name] = {'n': 0., 'N': 0, 'd': 0., 'Ne': 0, 'c': 0., 'SoS': 0}

    # update aggregates for online and baseline policies
    d[ts_bin]['online']['d'] += 1
    d[ts_bin]['online']['N'] += 1
    d[ts_bin]['baselineRand']['N'] += 1
    d[ts_bin]['baseline1']['N'] += 1

    d[ts_bin]['baselineRand']['Ne'] += 1
    d[ts_bin]['baselineRand']['d'] += 1/data['p']/data['num_a']
    if data['a'] == 1:
        d[ts_bin]['baseline1']['Ne'] += 1
        d[ts_bin]['baseline1']['d'] += 1/data['p']

    if r != 0:
        d[ts_bin]['online']['n'] += r
        d[ts_bin]['baselineRand']['n'] += r/data['p']/data['num_a']
        d[ts_bin]['baselineRand']['c'] = max(d[ts_bin]['baselineRand']['c'], abs_r/data['p']/data['num_a'])
        d[ts_bin]['baselineRand']['SoS'] += (r/data['p']/data['num_a'])**2
        if data['a'] == 1:
            d[ts_bin]['baseline1']['n'] += r/data['p']
            d[ts_bin]['baseline1']['c'] = max(d[ts_bin]['baseline1']['c'], abs_r/data['p'])
            d[ts_bin]['baseline1']['SoS'] += (r/data['p'])**2

    # update aggregates for additional policies from predictions
    for name in pred:
        # a-1: 0-index action
        pred_prob = get_prediction_prob(data['a'] - 1, pred[name][evts])
        d[ts_bin][name]['N'] += 1
        if pred_prob > 0:
            p_over_p = pred_prob/data['p']
            d[ts_bin][name]['d'] += p_over_p
            d[ts_bin][name]['Ne'] += 1
            if r != 0:
                d[ts_bin][name]['n'] += r*p_over_p
                d[ts_bin][name]['c'] = max(d[ts_bin][name]['c'], abs_r * p_over_p)
                d[ts_bin][name]['SoS'] += (r*p_over_p)**2
    return d


def aggregates_ccb_data(data, pred, d, evts):
    # binning timestamp every 5 min
    ts_bin = get_ts_5min_bin(data['ts'])

    # initialize aggregates for ts_bin
    if ts_bin not in d:
        d[ts_bin] = collections.OrderedDict([
            ('online', {'n': 0, 'N': 0, 'd': 0}),
            ('baseline1', {'n': 0., 'N': 0, 'd': 0., 'Ne': 0, 'c': 0., 'SoS': 0}),
            ('baselineRand', {'n': 0., 'N': 0, 'd': 0., 'Ne': 0, 'c': 0., 'SoS': 0})
        ])

        for name in pred:
            d[ts_bin][name] = {'n':0.,'N':0,'d':0.,'Ne':0,'c':0.,'SoS':0}

    # update aggregates for online and baseline policies
    d[ts_bin]['online']['d'] += len(data["_outcomes"])
    d[ts_bin]['online']['N'] += len(data["_outcomes"])
    d[ts_bin]['baseline1']['N'] += len(data["_outcomes"])

    d[ts_bin]['baselineRand']['N'] += len(data["_outcomes"])
    d[ts_bin]['baselineRand']['Ne'] += len(data["_outcomes"])

    for name in pred:
        d[ts_bin][name]['N'] += len(data["_outcomes"])

    for index, item in enumerate(data['_outcomes']):
        reward = -float(item["_label_cost"])
        abs_reward = abs(reward)

        d[ts_bin]['online']['n'] += reward

        if (item["_a"][0] == min(item["_a"])):
            d[ts_bin]['baseline1']['Ne'] += len(data["_outcomes"])
            d[ts_bin]['baseline1']['d'] += 1/item['_p'][0]
            d[ts_bin]['baseline1']['n'] += reward/item["_p"][0]
            d[ts_bin]['baseline1']['c'] = max(
                d[ts_bin]['baseline1']['c'],
                abs_reward / item["_p"][0]
            )
            d[ts_bin]['baseline1']['SoS'] += (reward/item["_p"][0])**2

        d[ts_bin]['baselineRand']['d'] += 1/item['_p'][0]/len(item['_a'])
        d[ts_bin]['baselineRand']['n'] += reward/item['_p'][0]/len(item['_a'])
        d[ts_bin]['baselineRand']['c'] = max(
            d[ts_bin]['baselineRand']['c'],
            abs_reward/item['_p'][0]/len(item['_a'])
        )
        d[ts_bin]['baselineRand']['SoS'] += (reward/item['_p'][0]/len(item['_a']))**2

        for name in pred:
            pred_prob = get_prediction_prob(item['_a'][0], pred[name][evts][index])
            if pred_prob > 0:
                p_over_p = pred_prob/item['_p'][0]
                d[ts_bin][name]['d'] += p_over_p
                d[ts_bin][name]['Ne'] += 1
                if reward != 0:
                    d[ts_bin][name]['n'] += reward*p_over_p
                    d[ts_bin][name]['c'] = max(
                        d[ts_bin][name]['c'],
                        abs_reward*p_over_p
                    )
                    d[ts_bin][name]['SoS'] += (reward*p_over_p)**2
    return d


def add_parser_args(parser):
    parser.add_argument('-l','--log_fp', help="data file path (.json or .json.gz format - each line is a dsjson)", required=True)
    parser.add_argument('-o','--output_fp', help="output file (default: log_fp+'.dash'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_parser_args(parser)
    args_dict = vars(parser.parse_args())   # this creates a dictionary with all input CLI
    for x in args_dict:
        locals()[x] = args_dict[x]  # this is equivalent to foo = args.foo

    if not output_fp:
        output_fp = log_fp+'.dash'

    d = create_stats(log_fp)
    output_dashboard_data(d, output_fp)
