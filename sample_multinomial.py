from src.multinomial import wrapper
import glob
import pandas as pd

for x in range(100):
    wrapper(
        iteration=x,
        experiment_key="06ee8e987b014a4d9b6b824ad6d28d83",
    )
