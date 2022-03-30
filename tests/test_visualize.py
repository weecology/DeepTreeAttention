#test validate
from src import visualize
def test_confusion_matrix(dm, rgb_pool, m, comet_logger, ROOT):
    if comet_logger:
        m.ROOT = "{}/tests".format(ROOT)
        results = m.evaluate_crowns(data_loader = dm.val_dataloader(), crowns=dm.crowns)
        visualize.confusion_matrix(
            comet_experiment=comet_logger.experiment,
            results=results,
            species_label_dict=dm.species_label_dict,
            test=dm.test,
            test_points=dm.canopy_points,
            test_crowns=dm.crowns,
            rgb_pool=rgb_pool)
    else:
        pass
    
def test_plot_spectra(m, dm, comet_logger):
    if comet_logger:
        experiment = comet_logger.experiment
    else:
        experiment = None    
    results = m.predict_dataloader(
        dm.val_dataloader())    
    visualize.plot_spectra(results, dm.data_dir)
    
def test_rgb_plots(m, dm, config, comet_logger):
    if comet_logger:
        experiment = comet_logger.experiment
    else:
        experiment = None 
    results = m.predict_dataloader(
        dm.val_dataloader())        
    visualize.rgb_plots(results, config=config, test_crowns=dm.crowns, test_points=dm.canopy_points, experiment=experiment)
