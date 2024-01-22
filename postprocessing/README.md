# Post-processing

There are a number of steps required to post-process the results after inference from a YOLO model, due to the required [preprocessing](../preprocessing/README.md) which removes spatial information from the dataset. To do this we must:

- Recombine patch predictions and convert pixel information to coordinates ([convert_results_to_shp](convert_results_to_shp.py))
- Filter and merge predictions ([merge_bboxes](../bbox/merge_bboxes.py))
- Calculate prediction metrics against ground-truth information ([process_results](process_results.py))

See the [quickstart](../end2end.ipynb) and [metrics notebook](metrics_notebook.ipynb) for example usage.