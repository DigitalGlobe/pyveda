"""
This function returns a mock gbdx-auth requests session with a dummy token.  You can optionally pass in a real token
if you want to actually make requests.
"""


import os
from oauthlib.oauth2 import LegacyApplicationClient
from requests_oauthlib import OAuth2Session
from gbdx_auth import gbdx_auth
import vcr


def force(r1, r2):
    return True


my_vcr = vcr.VCR()
my_vcr.register_matcher('force', force)
my_vcr.match_on = ['force']


def get_mock_gbdx_session(token='dummytoken'):
    s = OAuth2Session(client=LegacyApplicationClient('asdf'),
                      auto_refresh_url='fdsa',
                      auto_refresh_kwargs={'client_id': 'asdf',
                                           'client_secret': 'fdsa'})

    s.token = token
    s.access_token = token
    return s


if 'GBDX_MOCK' not in os.environ:
    conn = get_mock_gbdx_session(token='dummytoken')
else:
    conn = gbdx_auth.get_session()
