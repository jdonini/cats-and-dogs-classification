#!/bin/bash 
 
git add .
echo -e "Please write the commit message: \c"
read commit_message
git commit -m "$commit_message"
git push origin master
