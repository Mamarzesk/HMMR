echo "setting up minc"

source /opt/minc/1.9.18/minc-toolkit-config.sh

echo "running register_all_methods_regularizer.py"

python3 /code/grid_search_reg_spacing.py
