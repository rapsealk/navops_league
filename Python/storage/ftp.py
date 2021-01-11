#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
from hashlib import sha256

from pyftpdlib.authorizers import DummyAuthorizer, AuthenticationFailed
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer # , ThreadedFTPServer, MultiProcessFTPServer

FTP_HOST = '0.0.0.0'
FTP_PORT = 69
FTP_ANONYMOUS_DIRECTORY = os.path.join(os.getcwd(), 'anonymous')
FTP_AUTHENTICATED_DIRECTORY = os.path.join(os.getcwd(), 'authenticated')


def digest(message):
    if sys.version_info.major >= 3:
        message = message.encode("utf-8")
    return sha256(message).hexdigest()


class DummyMD5Authroizer(DummyAuthorizer):

    def validate_authentication(self, username, password, handler):
        print('valid:', username, password)
        if username in self.user_table:
            print('table:', self.user_table[username]['pwd'])
        password = digest(password)
        if self.user_table[username]['pwd'] != password:
            raise AuthenticationFailed


def main():
    authroizer = DummyMD5Authroizer()

    authroizer.add_user('admin', digest('1111'), FTP_AUTHENTICATED_DIRECTORY, perm='elradfmwMT')
    authroizer.add_user('user', digest('1111'), FTP_AUTHENTICATED_DIRECTORY, perm='elr')
    authroizer.add_anonymous(FTP_ANONYMOUS_DIRECTORY)

    handler = FTPHandler
    handler.banner = "pyftpdlib FTP server"
    handler.authorizer = authroizer
    handler.passive_ports = range(60000, 65535)

    address = (FTP_HOST, FTP_PORT)
    server = FTPServer(address, handler)

    server.max_cons = 256
    server.max_cons_per_ip = 5

    server.serve_forever()


if __name__ == "__main__":
    main()
