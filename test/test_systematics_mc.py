import pathlib, scipy
import roc_picker.systematics_mc

here = pathlib.Path(__file__).parent
docsfolder = here.parent/"docs"

responder_numerators_neighborhoods = [1093, 1734, 1740, 3557, 4163, 5679, 9852, 84076]
responder_numerators_cells = [4, 4, 15, 22, 0, 14, 24, 827]
responder_denominators = [6474, 37303, 28578, 16653, 21576, 28564, 46745, 564119]

nonresponder_numerators_neighborhoods = [47, 414, 523, 826, 945, 1156, 1176, 1297, 1773, 1863, 2075, 2138, 3621, 5012, 5684, 5828, 9175]
nonresponder_numerators_cells = [0, 13, 0, 0, 0, 3, 0, 5, 2, 0, 6, 2, 22, 18, 21, 6, 9]
nonresponder_denominators = [4129, 15941, 11636, 4325, 17332, 13985, 6388, 16170, 12316, 21463, 51148, 16397, 85422, 35832, 50968, 81808, 116977]

def plot_poisson(responder_numerators, responder_denominators, nonresponder_numerators, nonresponder_denominators, saveas=None, id_start=0):
  responder_num_distributions = [roc_picker.systematics_mc.ScipyDistribution(nominal=x, scipydistribution=scipy.stats.poisson(mu=x), id=i) for i, x in enumerate(responder_numerators, start=id_start)]
  responder_denom_distributions = [roc_picker.systematics_mc.ScipyDistribution(nominal=x, scipydistribution=scipy.stats.poisson(mu=x), id=i) for i, x in enumerate(responder_denominators, start=100+id_start)]
  nonresponder_num_distributions = [roc_picker.systematics_mc.ScipyDistribution(nominal=x, scipydistribution=scipy.stats.poisson(mu=x), id=i) for i, x in enumerate(nonresponder_numerators, start=200+id_start)]
  nonresponder_denom_distributions = [roc_picker.systematics_mc.ScipyDistribution(nominal=x, scipydistribution=scipy.stats.poisson(mu=x), id=i) for i, x in enumerate(nonresponder_denominators, start=300+id_start)]

  responders = [num / denom for num, denom in zip(responder_num_distributions, responder_denom_distributions, strict=True)]
  nonresponders = [num / denom for num, denom in zip(nonresponder_num_distributions, nonresponder_denom_distributions, strict=True)]

  rd = roc_picker.systematics_mc.ROCDistributions(responders=responders, nonresponders=nonresponders, flip_sign=True)
  rocs = rd.generate(size=10000, random_state=123456)
  rocs.plot(
    saveas=saveas,
  )

if __name__ == "__main__":
  plot_poisson(
    responder_numerators=responder_numerators_neighborhoods,
    responder_denominators=responder_denominators,
    nonresponder_numerators=nonresponder_numerators_neighborhoods,
    nonresponder_denominators=nonresponder_denominators,
    id_start=1000,
    saveas=docsfolder/"poisson_roc_neighborhoods.pdf",
  )

  plot_poisson(
    responder_numerators=responder_numerators_cells,
    responder_denominators=responder_denominators,
    nonresponder_numerators=nonresponder_numerators_cells,
    nonresponder_denominators=nonresponder_denominators,
    id_start=2000,
    saveas=docsfolder/"poisson_roc_cells.pdf",
  )
