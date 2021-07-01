#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os


def main():
    dotnet_installed = (os.system('dotnet --version') == 0)
    nodejs_installed = (os.system('node --version') == 0)

    if dotnet_installed:
        proj_path = os.path.join(os.path.dirname(__file__), 'NavOpsReplayGenApp', 'NavOpsReplayGenApp')
        cs_path = os.path.join(proj_path, 'Program.cs')
        print('dotnet path:', cs_path)
        os.system(f'dotnet run {cs_path} --project={proj_path} --path=hi')
    elif nodejs_installed:
        path = os.path.join(os.path.dirname(__file__), 'node', 'index.js')
        print('nodejs path:', path)
        os.system(f'node {path}')
    else:
        raise Exception()


if __name__ == "__main__":
    main()
