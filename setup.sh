#export PYTHONPATH=/data/drinkingkazu/summer2017:$PYTHONPATH
where="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$where:$PYTHONPATH
