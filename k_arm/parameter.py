
PARAMS = {
    'init_cost': 1e-03,  # 或 1e-03
    'steps': 1000,
    'round': 60,
    'lr': 1e-01,
    'attack_success_threshold': 0.99,
    'patience': 5,
    'channels': 3,
    'batch_size': 32,
    'single_pixel_trigger_optimization': True,
    'epsilon': 1e-07,
    'bandits_epsilon': 0.3,
    'beta': 1e+4,
    'warmup_rounds': 2,
    'cost_multiplier': 1.5,
    'early_stop': False,
    'early_stop_threshold': 1,
    'early_stop_patience': 10,
    'central_init': True,
    'universal_attack_trigger_size_bound': 1720,
    'label_specific_attack_trigger_size_bound': 1000,
    'symmetric_check_bound': 10,
}
