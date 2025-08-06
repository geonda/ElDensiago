from guess import MlDensity

predictor = MlDensity(
            model='nmc_schnet',
            device='cpu',
            grid_step=0.5,
            probe_count=10,
            force_pbc=True
        )
predictor.predict("example/batio3.xyz")