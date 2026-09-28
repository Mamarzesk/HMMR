echo "setting up minc"

source /opt/minc/1.9.18/minc-toolkit-config.sh

echo "running register_sample_size.py"

python3 /code/register_sample_size.py
