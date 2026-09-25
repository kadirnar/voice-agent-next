"""Amazon Web Services provider package (Amazon Bedrock).

Submodules register their components on import:

* ``nova_sonic`` — the Amazon Nova 2 Sonic speech-to-speech engine
  (``InvokeModelWithBidirectionalStream``)
"""

from __future__ import annotations

from . import nova_sonic  # noqa: F401  (import registers the components)
