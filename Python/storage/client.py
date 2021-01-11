#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from ftplib import FTP

FTP_HOST = 'localhost'
FTP_PORT = 69


def main():
    # ftp = FTP(source_address=('localhost', 6969))
    ftp = FTP()
    ftp.connect(FTP_HOST, FTP_PORT)
    ftp.login(user='admin', passwd='1111')
    # ftp.cwd('../authenticated')
    ftp.retrlines('LIST')

    with open('downloaded.md', 'wb') as fp:
        ftp.retrbinary('RETR README.md', fp.write)

    ftp.quit()


if __name__ == "__main__":
    main()
