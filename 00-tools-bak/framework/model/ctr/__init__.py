#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import hydra


@hydra.main()
def app(cfg):
    print(cfg.pretty())
    print("The user is : " + cfg.user)


if __name__ == "__main__":
    app()