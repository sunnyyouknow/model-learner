cd /diskc/model-learner-maiya/model-learner/code
echo $1
echo $2 
python -W ignore feature_test.py $1 $2
