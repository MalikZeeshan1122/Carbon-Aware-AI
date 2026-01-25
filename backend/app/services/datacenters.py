# Data Center Database with Carbon Emissions Information
# Sources: Google Environmental Reports, AWS Sustainability, Microsoft Carbon Reports, IEA

DATACENTER_INFO = {
    # --- Google Cloud ---
    "gcp-us-central1": {
        "name": "Google Cloud - Iowa",
        "provider": "Google",
        "location": "Council Bluffs, Iowa, USA",
        "carbon_intensity": 410,  # gCO2/kWh
        "pue": 1.10,
        "renewable_percent": 100,
        "cooling": "Free Cooling + Evaporative",
        "capacity_mw": 400,
        "annual_emissions_tonnes": 0,  # Carbon neutral since 2017
        "notes": "Google matches 100% with renewables. Effective CI near 0."
    },
    "gcp-us-west1": {
        "name": "Google Cloud - Oregon",
        "provider": "Google",
        "location": "The Dalles, Oregon, USA",
        "carbon_intensity": 80,
        "pue": 1.08,
        "renewable_percent": 100,
        "cooling": "Free Cooling (Columbia River)",
        "capacity_mw": 500,
        "annual_emissions_tonnes": 0,
        "notes": "One of the most efficient DCs in the world. Hydro-powered region."
    },
    "gcp-europe-west1": {
        "name": "Google Cloud - Belgium",
        "provider": "Google",
        "location": "St. Ghislain, Belgium",
        "carbon_intensity": 150,
        "pue": 1.09,
        "renewable_percent": 100,
        "cooling": "Industrial Canal Water",
        "capacity_mw": 300,
        "annual_emissions_tonnes": 0,
        "notes": "First DC to operate without chillers."
    },
    
    # --- AWS ---
    "aws-us-east-1": {
        "name": "AWS US East (N. Virginia)",
        "provider": "AWS",
        "location": "Ashburn, Virginia, USA",
        "carbon_intensity": 340,
        "pue": 1.20,
        "renewable_percent": 85,
        "cooling": "Mechanical + Evaporative",
        "capacity_mw": 1500,
        "annual_emissions_tonnes": 450000,
        "notes": "Largest AWS region. Heavy enterprise workloads."
    },
    "aws-us-west-2": {
        "name": "AWS US West (Oregon)",
        "provider": "AWS",
        "location": "Boardman, Oregon, USA",
        "carbon_intensity": 100,
        "pue": 1.15,
        "renewable_percent": 95,
        "cooling": "Free Cooling",
        "capacity_mw": 800,
        "annual_emissions_tonnes": 50000,
        "notes": "Powered largely by Columbia River hydro."
    },
    "aws-eu-west-1": {
        "name": "AWS Europe (Ireland)",
        "provider": "AWS",
        "location": "Dublin, Ireland",
        "carbon_intensity": 280,
        "pue": 1.16,
        "renewable_percent": 90,
        "cooling": "Free Cooling (Atlantic climate)",
        "capacity_mw": 600,
        "annual_emissions_tonnes": 120000,
        "notes": "Cool climate allows efficient operations."
    },
    "aws-ap-southeast-1": {
        "name": "AWS Asia Pacific (Singapore)",
        "provider": "AWS",
        "location": "Singapore",
        "carbon_intensity": 420,
        "pue": 1.30,
        "renewable_percent": 40,
        "cooling": "Mechanical (Tropical)",
        "capacity_mw": 400,
        "annual_emissions_tonnes": 180000,
        "notes": "Tropical climate requires heavy cooling."
    },
    
    # --- Microsoft Azure ---
    "azure-eastus": {
        "name": "Azure East US",
        "provider": "Microsoft",
        "location": "Virginia, USA",
        "carbon_intensity": 350,
        "pue": 1.18,
        "renewable_percent": 80,
        "cooling": "Hybrid Cooling",
        "capacity_mw": 1000,
        "annual_emissions_tonnes": 350000,
        "notes": "Major enterprise hub. OpenAI workloads here."
    },
    "azure-westeurope": {
        "name": "Azure West Europe",
        "provider": "Microsoft",
        "location": "Amsterdam, Netherlands",
        "carbon_intensity": 200,
        "pue": 1.12,
        "renewable_percent": 100,
        "cooling": "Free Cooling + Canal Water",
        "capacity_mw": 500,
        "annual_emissions_tonnes": 80000,
        "notes": "100% renewable matched. Efficient cooling."
    },
    "azure-northeurope": {
        "name": "Azure North Europe",
        "provider": "Microsoft",
        "location": "Dublin, Ireland",
        "carbon_intensity": 280,
        "pue": 1.14,
        "renewable_percent": 95,
        "cooling": "Free Cooling",
        "capacity_mw": 450,
        "annual_emissions_tonnes": 100000,
        "notes": "Atlantic climate beneficial for cooling."
    },
    "azure-swedencentral": {
        "name": "Azure Sweden Central",
        "provider": "Microsoft",
        "location": "Gävle/Sandviken, Sweden",
        "carbon_intensity": 20,
        "pue": 1.08,
        "renewable_percent": 100,
        "cooling": "Free Cooling (Nordic)",
        "capacity_mw": 300,
        "annual_emissions_tonnes": 5000,
        "notes": "Among cleanest DCs globally. Nuclear + Hydro grid."
    },
    
    # --- Independent/Other ---
    "equinix-sv5": {
        "name": "Equinix SV5",
        "provider": "Equinix",
        "location": "San Jose, California, USA",
        "carbon_intensity": 180,
        "pue": 1.40,
        "renewable_percent": 100,
        "cooling": "Mechanical",
        "capacity_mw": 15,
        "annual_emissions_tonnes": 8000,
        "notes": "High PUE but 100% renewable energy contracts."
    },
    "coresite-la1": {
        "name": "CoreSite LA1",
        "provider": "CoreSite",
        "location": "Los Angeles, California, USA",
        "carbon_intensity": 200,
        "pue": 1.35,
        "renewable_percent": 60,
        "cooling": "Mechanical + Evaporative",
        "capacity_mw": 20,
        "annual_emissions_tonnes": 15000,
        "notes": "Colocation facility, mixed grid."
    }
}

# Aggregated regional data for simplified lookups
REGION_SUMMARY = {
    "us-west": {"avg_ci": 120, "best_dc": "gcp-us-west1", "worst_dc": "equinix-sv5"},
    "us-east": {"avg_ci": 360, "best_dc": "gcp-us-central1", "worst_dc": "aws-us-east-1"},
    "eu-west": {"avg_ci": 240, "best_dc": "azure-westeurope", "worst_dc": "aws-eu-west-1"},
    "eu-north": {"avg_ci": 30, "best_dc": "azure-swedencentral", "worst_dc": None},
    "asia-pacific": {"avg_ci": 500, "best_dc": "aws-ap-southeast-1", "worst_dc": "aws-ap-southeast-1"}
}

def get_all_datacenters():
    """Return all data centers with their info."""
    return DATACENTER_INFO

def get_datacenter(dc_id: str) -> dict:
    """Get info for a specific data center."""
    return DATACENTER_INFO.get(dc_id, None)

def get_datacenters_by_provider(provider: str) -> dict:
    """Get all data centers for a specific provider."""
    return {k: v for k, v in DATACENTER_INFO.items() if v['provider'].lower() == provider.lower()}

def get_cleanest_datacenters(top_n: int = 5) -> list:
    """Get the top N cleanest data centers by carbon intensity."""
    sorted_dcs = sorted(DATACENTER_INFO.items(), key=lambda x: x[1]['carbon_intensity'])
    return sorted_dcs[:top_n]

def get_datacenter_comparison() -> list:
    """Get comparison data for visualization."""
    return [
        {
            "id": dc_id,
            "name": info["name"],
            "provider": info["provider"],
            "carbon_intensity": info["carbon_intensity"],
            "pue": info["pue"],
            "renewable_percent": info["renewable_percent"],
            "annual_emissions_tonnes": info["annual_emissions_tonnes"]
        }
        for dc_id, info in DATACENTER_INFO.items()
    ]
