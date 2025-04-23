echo "checking python imports"

python3 /code/check_imports.py

echo "listing data directory. BITE and RESECT should be present"

ls /data

source /opt/minc/1.9.18/minc-toolkit-config.sh

echo "checking minc toolkit"

if mincinfo -version > /dev/null; then
    echo "minc toolkit properly enabled"
else
    echo "minc toolkit not enabled. failure"
    exit 1
fi

exit 0
