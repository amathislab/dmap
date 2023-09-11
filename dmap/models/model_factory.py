from ray.rllib.models import ModelCatalog


class ModelFactory:
    """Static factory to register models from strings

    Raises:
        ValueError: when the input string does not correspond to any known model
    """

    @staticmethod
    def register(model_name):
        if model_name == "oracle_policy":
            from dmap.models.rma import OraclePolicyModel

            ModelCatalog.register_custom_model(model_name, OraclePolicyModel)
        elif model_name == "oracle_q":
            from dmap.models.rma import OracleQModel

            ModelCatalog.register_custom_model(model_name, OracleQModel)
        elif model_name == "tcn_policy":
            from dmap.models.rma import TCNPolicyModel

            ModelCatalog.register_custom_model(model_name, TCNPolicyModel)
        elif model_name == "oracle_q_adapt":
            from dmap.models.rma import OracleQAdaptModel

            ModelCatalog.register_custom_model(model_name, OracleQAdaptModel)
        elif model_name == "tcn_q":
            from dmap.models.rma import TCNQModel

            ModelCatalog.register_custom_model(model_name, TCNQModel)
        elif model_name == "dmap_policy":
            from dmap.models.dmap import DMAPPolicyModel

            ModelCatalog.register_custom_model(model_name, DMAPPolicyModel)
        elif model_name == "dmap_icl":
            from dmap.models.dmap_icl import DMAPPolicyModelICL

            ModelCatalog.register_custom_model(model_name, DMAPPolicyModelICL)
        else:
            raise ValueError("Unknown model name", model_name)

    @staticmethod
    def register_models_from_config(policy_configs):
        for policy in policy_configs.values():
            for model_params in policy.values():
                if isinstance(model_params, dict):
                    model_name = model_params.get("custom_model")
                    if model_name is not None:
                        ModelFactory.register(model_name)
