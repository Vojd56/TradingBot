import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TickerMapper:
    """
    Zajišťuje obousměrný převod mezi standardními burzovními tickery (např. AAPL, NVDA)
    a interními symboly Trading 212 Invest API (např. AAPL_US_EQ, NVDA_US_EQ).
    """

    # Manuální výjimky pro tickery se speciálními znaky a evropské burzy
    SPECIAL_MAPPINGS: Dict[str, str] = {
        "BRK-B": "BRK_B_US_EQ",
        "BRK.B": "BRK_B_US_EQ",
        "BF-B": "BFb_US_EQ",
        "BF.B": "BFb_US_EQ",
        # Speciální a historické symboly na Trading 212
        "META": "FB_US_EQ",
        "SONY": "SNE_US_EQ",
        "RTX": "UTX_US_EQ",
        "SE": "GARENA_US_EQ",
        "HEI": "HEICO_US_EQ",
        # Evropské tituly na Trading 212
        "SUNB": "AHTl_EQ",
        "SUNB.L": "AHTl_EQ",
        "ABBN.SW": "ABBNsEQ",
        "ABBN": "ABBNsEQ",
        "RO.SW": "ROGs_EQ",
        "ROG.SW": "ROGs_EQ",
        "RO": "ROGs_EQ",
        "RHM.DE": "RHMd_EQ",
        "RHM": "RHMd_EQ",
        "RR.L": "RRl_EQ",
        "RYCEY": "RYCEY_US_EQ",
        "SMSN.IL": "SMSNl_EQ",
        "SMSN.L": "SMSNl_EQ",
        "SMSN": "SMSNl_EQ",
        "SAP.DE": "SAPd_EQ",
        "NOVO-B.CO": "NOVCd_EQ",
        "NESN.SW": "NESNs_EQ",
        "NESN": "NESNs_EQ",
        "ASML.AS": "ASMLa_EQ",
        "TTE.PA": "FPp_EQ",
        "TTE": "FPp_EQ",
        "AIR.PA": "AIRp_EQ",
        "AIR": "AIRp_EQ",
        "RIO.L": "RIOl_EQ",
        "RIO": "RIOl_EQ",
    }

    REVERSE_SPECIAL: Dict[str, str] = {
        "BRK_B_US_EQ": "BRK-B",
        "BRKb_US_EQ": "BRK-B",
        "BFb_US_EQ": "BF-B",
        "FB_US_EQ": "META",
        "SNE_US_EQ": "SONY",
        "UTX_US_EQ": "RTX",
        "GARENA_US_EQ": "SE",
        "HEICO_US_EQ": "HEI",
        "AHTl_EQ": "SUNB",
        "ABBNsEQ": "ABBN.SW",
        "ROGs_EQ": "RO.SW",
        "FPp_EQ": "TTE.PA",
        "TTEp_EQ": "TTE.PA",
        "RHMd_EQ": "RHM.DE",
        "RRl_EQ": "RR.L",
        "RYCEY_US_EQ": "RYCEY",
        "SMSNl_EQ": "SMSN.IL",
        "SAPd_EQ": "SAP.DE",
        "NOVCd_EQ": "NOVO-B.CO",
        "NESNs_EQ": "NESN.SW",
        "ASMLa_EQ": "ASML.AS",
        "AIRp_EQ": "AIR.PA",
        "RIOl_EQ": "RIO.L",
    }

    @classmethod
    def to_t212_ticker(cls, standard_ticker: str) -> str:
        """
        Převede standardní ticker na T212 symbol.
        Např.: 'NVDA' -> 'NVDA_US_EQ', 'BRK-B' -> 'BRKb_US_EQ'.
        """
        raw = standard_ticker.strip()
        # Pokud již obsahuje příponu, vrátíme beze změny (zachováme původní velikost písmen jako SMSNl_EQ či ABBNsEQ)
        if raw.endswith("_US_EQ") or raw.endswith("_EQ") or raw == "ABBNsEQ":
            return raw

        clean = raw.upper()
        if clean in cls.SPECIAL_MAPPINGS:
            return cls.SPECIAL_MAPPINGS[clean]

        # Automatické mapování evropských burzovních koncovek na Trading 212 formát
        if clean.endswith(".DE"):
            return f"{clean[:-3]}d_EQ"
        if clean.endswith(".L"):
            return f"{clean[:-2]}l_EQ"
        if clean.endswith(".PA"):
            return f"{clean[:-3]}p_EQ"
        if clean.endswith(".AS"):
            return f"{clean[:-3]}a_EQ"
        if clean.endswith(".SW"):
            return f"{clean[:-3]}s_EQ"

        return f"{clean}_US_EQ"

    @classmethod
    def to_standard_ticker(cls, t212_ticker: str) -> str:
        """
        Převede T212 symbol zpět na standardní burzovní ticker.
        Např.: 'NVDA_US_EQ' -> 'NVDA', 'AIRp_EQ' -> 'AIR.PA'.
        """
        clean = t212_ticker.strip()
        if clean in cls.REVERSE_SPECIAL:
            return cls.REVERSE_SPECIAL[clean]

        if clean.endswith("_US_EQ"):
            return clean[:-6]
        if clean.endswith("d_EQ"):
            return f"{clean[:-4]}.DE"
        if clean.endswith("l_EQ"):
            return f"{clean[:-4]}.L"
        if clean.endswith("p_EQ"):
            return f"{clean[:-4]}.PA"
        if clean.endswith("a_EQ"):
            return f"{clean[:-4]}.AS"
        if clean.endswith("s_EQ"):
            return f"{clean[:-4]}.SW"
        if clean.endswith("_EQ"):
            return clean.split("_")[0]

        return clean
