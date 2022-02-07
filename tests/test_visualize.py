#test validate
from src import visualize
def test_confusion_matrix(dm, rgb_pool, m, experiment, ROOT):
    if experiment:
        m.ROOT = "{}/tests".format(ROOT)
        results = m.evaluate_crowns(data_loader = dm.val_dataloader())
        visualize.confusion_matrix(
            comet_experiment=experiment,
            results=results,
            species_label_dict=dm.species_label_dict,
            test_csv="{}/tests/data/processed/test.csv".format(ROOT),
            test_points="{}/tests/data/processed/test_points.shp".format(ROOT),
            test_crowns="{}/tests/data/processed/test_crowns.shp".format(ROOT),
            rgb_pool=rgb_pool)
    else:
        pass