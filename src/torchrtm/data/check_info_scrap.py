import pickle
import pprint
from pathlib import Path

pp = pprint.PrettyPrinter()
script_dir = Path(__file__).parent.resolve()

# sensor_information 在脚本所在目录下
sensor_dir = script_dir / "sensor_information"
def get_rs_info(show_detail = False):
    names = []
    for sensor in sensor_dir.glob("*.pkl"):
        print(sensor.name)
        if show_detail:
            with open(sensor, "rb") as f:
                info = pickle.load(f)
                
                pp.pprint(info.keys())
        names.append(sensor.name)
    return(names)