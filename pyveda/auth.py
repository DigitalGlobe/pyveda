import os
import requests
from requests.adapters import HTTPAdapter
from gbdx_auth import gbdx_auth
import logging

HOST = os.environ.get('SANDMAN_API', "https://veda-api.geobigdata.io")

auth = None

def Auth(**kwargs):
    global auth
    if auth is None or len(kwargs) > 0:
        auth = _Auth(**kwargs)
    return auth


class _Auth(object):
    gbdx_connection = None
    root_url = HOST

    def __init__(self, **kwargs):
        self.logger = logging.getLogger('pyveda')
        self.logger.setLevel(logging.ERROR)
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.ERROR)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.console_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.console_handler)
        self.logger.info('Logger initialized')

        if 'host' in kwargs:
            self.root_url = 'https://%s' % kwargs.get('host')

        if (kwargs.get('username') and kwargs.get('password')):
            self.gbdx_connection = gbdx_auth.session_from_kwargs(**kwargs)
        elif kwargs.get('gbdx_connection'):
            self.gbdx_connection = kwargs.get('gbdx_connection')
        elif self.gbdx_connection is None:
            # This will throw an exception if your .ini file is not set properly
            self.gbdx_connection = gbdx_auth.get_session(kwargs.get('config_file'))

        # for local dev, cant use oauth2
        if HOST == 'http://host.docker.internal:3002':
            headers = {"Authorization": "Bearer {}".format(self.gbdx_connection.access_token)}
            self.gbdx_connection = requests.Session()
            self.gbdx_connection.headers.update(headers) 

        def expire_token(r, *args, **kw):
            """
            Requests a new token if 401, retries request, mainly for auth v2 migration
            :param r:
            :param args:
            :param kw:
            :return:
            """
            if r.status_code == 401:
                try:
                    # remove hooks so it doesn't get into infinite loop
                    r.request.hooks = None
                    # expire the token
                    gbdx_auth.expire_token(token_to_expire=self.gbdx_connection.token,
                                           config_file=kwargs.get('config_file'))
                    # re-init the session
                    self.gbdx_connection = gbdx_auth.get_session(kwargs.get('config_file'))

                    # make original request, triggers new token request first
                    res = self.gbdx_connection.request(method=r.request.method, url=r.request.url)

                    # re-add the hook to refresh in the future 
                    self.gbdx_connection.hooks['response'].append(expire_token)
                    return res  
                  

                except Exception as e:
                    r.request.hooks = None
                    print("Error expiring token from session, Reason {}".format(e))

        if self.gbdx_connection is not None:
            self.gbdx_connection.hooks['response'].append(expire_token)
