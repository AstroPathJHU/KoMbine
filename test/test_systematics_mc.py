import functools, pathlib, scipy
import roc_picker.systematics_mc

here = pathlib.Path(__file__).parent
docsfolder = here.parent/"docs"

responder_numerators_neighborhoods = [1093, 1734, 1740, 3557, 4163, 5679, 9852, 84076]
responder_numerators_cells = [4, 4, 15, 22, 0, 14, 24, 827]
responder_denominators = [6474, 37303, 28578, 16653, 21576, 28564, 46745, 564119]
responder_primaries = [355, 1257, 338, 1514, 2668, 2723, 2940, 108390]
responder_secondaries = [337, 1233, 305, 1435, 2500, 2752, 2820, 107509]
responder_batchids = [2, 2, 6, 1, 8, 1, 3, 3]

nonresponder_numerators_neighborhoods = [47, 414, 523, 826, 945, 1156, 1176, 1297, 1773, 1863, 2075, 2138, 3621, 5012, 5684, 5828, 9175]
nonresponder_numerators_cells = [0, 13, 0, 0, 0, 3, 0, 5, 2, 0, 6, 2, 22, 18, 21, 6, 9]
nonresponder_denominators = [4129, 15941, 11636, 4325, 17332, 13985, 6388, 16170, 12316, 21463, 51148, 16397, 85422, 35832, 50968, 81808, 116977]
nonresponder_primaries = [234, 204, 285, 386, 458, 393, 358, 372, 784, 799, 778, 1300, 2356, 1988, 2852, 1379, 5856]
nonresponder_secondaries = [173, 199, 253, 388, 416, 404, 389, 400, 720, 826, 789, 1282, 2121, 1971, 2592, 1423, 5144]
nonresponder_batchids = [12, 11, 7, 4, 13, 2, 12, 6, 2, 4, 2, 10, 8, 3, 1, 4, 1]

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

def plot_lognormal(responder_numerators, responder_denominators, responder_primaries, responder_secondaries, responder_batchids, nonresponder_numerators, nonresponder_denominators, nonresponder_primaries, nonresponder_secondaries, nonresponder_batchids, saveas=None, id_start=0):
  @functools.cache
  def systematic(batchid):
    num = denom = 0
    for (primaries, secondaries, batchids) in (
      (responder_primaries, responder_secondaries, responder_batchids),
      (nonresponder_primaries, nonresponder_secondaries, nonresponder_batchids),
    ):
      for p, s, b in zip(primaries, secondaries, batchids, strict=True):
        if b == batchid:
          num += s
          denom += p

    onesigma = num/denom
    gaussian = roc_picker.systematics_mc.ScipyDistribution(nominal=0, scipydistribution=scipy.stats.norm(), id=id_start+batchid)
    return onesigma ** gaussian

  responders = [(num / denom) * systematic(batchid) for num, denom, batchid in zip(responder_numerators, responder_denominators, responder_batchids, strict=True)]
  nonresponders = [(num / denom) * systematic(batchid) for num, denom, batchid in zip(nonresponder_numerators, nonresponder_denominators, nonresponder_batchids, strict=True)]

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

  plot_lognormal(
    responder_numerators=responder_numerators_neighborhoods,
    responder_denominators=responder_denominators,
    responder_batchids=responder_batchids,
    responder_primaries=responder_primaries,
    responder_secondaries=responder_secondaries,
    nonresponder_numerators=nonresponder_numerators_neighborhoods,
    nonresponder_denominators=nonresponder_denominators,
    nonresponder_batchids=nonresponder_batchids,
    nonresponder_primaries=nonresponder_primaries,
    nonresponder_secondaries=nonresponder_secondaries,
    id_start=3000,
    saveas=docsfolder/"lognormal_roc.pdf",
  )
