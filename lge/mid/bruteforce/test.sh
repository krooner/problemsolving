#!/bin/zsh

test () {
    mkdir "$1"
    cd "$1"
    code answer.py
}