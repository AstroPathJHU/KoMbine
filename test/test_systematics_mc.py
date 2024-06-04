import pathlib, scipy
import roc_picker.systematics_mc

here = pathlib.Path(__file__).parent
docsfolder = here.parent/"docs"

responder_numerators = [1093, 1734, 1740, 3557, 4163, 5679, 9852, 84076]
responder_denominators = [6474, 37303, 28578, 16653, 21576, 28564, 46745, 564119]

nonresponder_numerators = [47, 414, 523, 826, 945, 1156, 1176, 1297, 1773, 1863, 2075, 2138, 3621, 5012, 5684, 5828, 9175]
nonresponder_denominators = [4129, 15941, 11636, 4325, 17332, 13985, 6388, 16170, 12316, 21463, 51148, 16397, 85422, 35832, 50968, 81808, 116977]

def plot_poisson(responder_numerators, responder_denominators, nonresponder_numerators, nonresponder_denominators):
  responder_num_distributions = [roc_picker.systematics_mc.ScipyDistribution(nominal=x, scipydistribution=scipy.stats.poisson(mu=x), id=i) for i, x in enumerate(responder_numerators)]
  responder_denom_distributions = [roc_picker.systematics_mc.ScipyDistribution(nominal=x, scipydistribution=scipy.stats.poisson(mu=x), id=i) for i, x in enumerate(responder_denominators, start=100)]
  nonresponder_num_distributions = [roc_picker.systematics_mc.ScipyDistribution(nominal=x, scipydistribution=scipy.stats.poisson(mu=x), id=i) for i, x in enumerate(nonresponder_numerators, start=200)]
  nonresponder_denom_distributions = [roc_picker.systematics_mc.ScipyDistribution(nominal=x, scipydistribution=scipy.stats.poisson(mu=x), id=i) for i, x in enumerate(nonresponder_denominators, start=300)]

  responders = [num / denom for num, denom in zip(responder_num_distributions, responder_denom_distributions, strict=True)]
  nonresponders = [num / denom for num, denom in zip(nonresponder_num_distributions, nonresponder_denom_distributions, strict=True)]

  rd = roc_picker.systematics_mc.ROCDistributions(responders=responders, nonresponders=nonresponders)
  rocs = rd.generate(size=1000, random_state=123456)
  rocs.plot(
    saveas=docsfolder/"poisson_roc.pdf",
  )

if __name__ == "__main__":
  plot_poisson(
    responder_numerators=responder_numerators,
    responder_denominators=responder_denominators,
    nonresponder_numerators=nonresponder_numerators,
    nonresponder_denominators=nonresponder_denominators,
  )

