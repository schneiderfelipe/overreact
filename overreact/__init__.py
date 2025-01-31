""".. include:: ../README.md"""  # noqa: D400

from __future__ import annotations

__docformat__ = "restructuredtext"

from importlib.metadata import version

from overreact.api import (
    get_enthalpies,
    get_entropies,
    get_freeenergies,
    get_internal_energies,
    get_k,
    get_kappa,
)
from overreact.core import (
    Scheme,
    get_transition_states,
    is_transition_state,
    parse_reactions,
    unparse_reactions,
)
from overreact.io import parse_compounds, parse_model
from overreact.simulate import get_bias, get_dydt, get_fixed_scheme, get_y
from overreact.thermo import change_reference_state, get_delta, get_reaction_entropies

__all__ = [
    "Scheme",
    "change_reference_state",
    "get_bias",
    "get_delta",
    "get_dydt",
    "get_enthalpies",
    "get_entropies",
    "get_fixed_scheme",
    "get_freeenergies",
    "get_internal_energies",
    "get_k",
    "get_kappa",
    "get_reaction_entropies",
    "get_transition_states",
    "get_y",
    "is_transition_state",
    "parse_compounds",
    "parse_model",
    "parse_reactions",
    "unparse_reactions",
]

__version__ = version(__name__)
__license__ = "MIT"  # I'm too lazy to get it from setup.py...

__headline__ = "📈 Create and analyze chemical microkinetic models built from computational chemistry data."

__url_repo__ = "https://github.com/geem-lab/overreact"
__url_issues__ = f"{__url_repo__}/issues"
__url_discussions__ = f"{__url_repo__}/discussions"
__url_pypi__ = "https://pypi.org/project/overreact/"
__url_guide__ = "https://geem-lab.github.io/overreact-guide/"

__doi__ = "10.1002/jcc.26861"
__zenodo_doi__ = "10.5281/zenodo.7865357"
__citations__ = (  # TODO(schneiderfelipe): read from CITATION.bib
    r"""
    @article{overreact_paper2022,
  title         = {Overreact, an in silico lab: Automative quantum chemical microkinetic simulations for complex chemical reactions},
  author        = {Schneider, Felipe S. S. and Caramori, Giovanni F.},
  year          = {2022},
  month         = {Apr},
  journal       = {Journal of Computational Chemistry},
  publisher     = {Wiley},
  volume        = {44},
  number        = {3},
  pages         = {209-217},
  doi           = {DOI_PLACEHOLDER},
  issn          = {1096-987x},
  url           = {http://dx.doi.org/DOI_PLACEHOLDER},
}
@software{overreact_software2021,
  title         = {geem-lab/overreact: vVERSION_PLACEHOLDER \vert{} Zenodo},
  author        = {Felipe S. S. Schneider and Let\'{\i}cia M. P. Madureira and  Giovanni F. Caramori},
  year          = {2023},
  month         = {Jan},
  publisher     = {Zenodo},
  doi           = {ZENODO_DOI_PLACEHOLDER},
  url           = {https://doi.org/ZENODO_DOI_PLACEHOLDER},
  version       = {vVERSION_PLACEHOLDER},
  howpublished  = {\url{URL_REPO_PLACEHOLDER}},
}
""".replace(
        "ZENODO_DOI_PLACEHOLDER",
        __zenodo_doi__,
    )
    .replace("DOI_PLACEHOLDER", __doi__)
    .replace("URL_REPO_PLACEHOLDER", __url_repo__)
    .replace("VERSION_PLACEHOLDER", __version__)
)
