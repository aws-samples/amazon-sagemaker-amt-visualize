# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from datetime import datetime, timedelta, timezone

import hashlib
import traceback
import os
from pathlib import Path

import pandas as pd
import numpy as np
import boto3

cw = boto3.client('cloudwatch')
sm = boto3.client('sagemaker')


def disk_cache(outer):
    def inner(*args, **kwargs):

        key_input = str(args)+str(kwargs)
        key = hashlib.md5(key_input.encode('utf-8')).hexdigest() # nosec b303 - Not used for cryptography, but to create lookup key 
        cache_dir = '.cache/cw_metrics/' 
        fn = f'{cache_dir}/req_{key}.jsonl.gz'
        if Path(fn).exists():
            try:
                df = pd.read_json(fn, lines=True)
                print('H', end='')
                df['ts'] = np.array(df['ts'], dtype=np.datetime64)
                df['rel_ts'] = np.array(df['rel_ts'], dtype=np.datetime64)
                return df
            except KeyError as e:
                pass # Empty file leads to empty df, hence no df['ts'] possible
            except BaseException as e: # nosec b110 - doesn't matter why we could not load it.
                print('\nException', type(e), e)
                pass # continue with calling the outer function 
         
        print('M', end='')
        df = outer(*args, **kwargs)
        assert(isinstance(df, pd.core.frame.DataFrame), 'Only caching Pandas DataFrames.')
        
        os.makedirs(cache_dir, exist_ok=True)
        df.to_json(fn, orient='records', date_format='iso', lines=True)

        return df 
    return inner

def _metric_data_query_tpl(metric_name, dim_name, dim_value):
    return {
      'Id': metric_name.lower().replace(':', '_').replace('-', '_'),
      'MetricStat': {
          'Stat': 'Average',
          'Metric': {
              'Namespace': '/aws/sagemaker/TrainingJobs',
              'MetricName': metric_name,
              'Dimensions': [
                  {
                      'Name': dim_name,
                      'Value': dim_value
                  },
              ]
          },
          'Period': 60,
      },
      'ReturnData':True,
  }

def _get_metric_data(queries, start_time, end_time):
    start_time = start_time - timedelta(hours=1)
    end_time = end_time + timedelta(hours=1)
    response = cw.get_metric_data(MetricDataQueries=queries,
      StartTime=start_time,
      EndTime=end_time)
    
    df = pd.DataFrame()
    for metric_data in response['MetricDataResults']:
        values = metric_data['Values']
        ts     = np.array(metric_data['Timestamps'], dtype=np.datetime64)
        labels = [metric_data['Label']]*len(values)
    
        df = pd.concat([df, pd.DataFrame({'value': values, 'ts': ts, 'label': labels})])
   
    
    # We now calculate the relative time based on the first actual observed time stamps, not
    # the potentially start time that we used to scope our CW API call. The difference
    # could be for example startup times or waiting for Spot.
    if not df.empty:
        df['rel_ts'] = datetime.fromtimestamp(0)+(df['ts']-df['ts'].min())
    return df

@disk_cache
def _collect_metrics(dimensions, start_time, end_time):
    
    if not end_time:
        end_time = start_time + timedelta(hours=4) 
    
    df = pd.DataFrame()
    for dim_name, dim_value in dimensions:
        response = cw.list_metrics(
            Namespace='/aws/sagemaker/TrainingJobs',
            Dimensions=[
                {
                    'Name': dim_name,
                    'Value': dim_value
                },
            ])
        metric_names = [metric['MetricName'] for metric in response['Metrics']]
        if not metric_names:
            # No metric data yet, or not any longer, because the data were aged out
            continue
        metric_data_queries = [_metric_data_query_tpl(metric_name, dim_name, dim_value) for metric_name in metric_names]
        df = pd.concat([df, _get_metric_data(metric_data_queries, start_time, end_time)])

    return df

def get_cw_job_metrics(job_name, 
                    start_time=None, 
                    end_time=None):
    dimensions = [('TrainingJobName', job_name), 
              ('Host', job_name+'/algo-1')]
    return _collect_metrics(dimensions, start_time, end_time)
