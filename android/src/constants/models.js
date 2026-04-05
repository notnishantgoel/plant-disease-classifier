// Change this to your machine's local IP when testing on a real device
// e.g. 'http://192.168.1.42:8000'
export const API_BASE_URL = 'http://10.0.2.2:8000' // Android emulator → host machine

export const MODELS = [
  {
    id: 'vit_moe_v3',
    name: 'Vision Transformer + MoE',
    shortName: 'ViT + MoE',
    accuracy: '91.125%',
    icon: '🧠',
    description: 'Custom ViT with Mixture-of-Experts routing. Best accuracy.',
  },
  {
    id: 'efficientnet',
    name: 'EfficientNet v2-S',
    shortName: 'EfficientNet',
    accuracy: '94.86%',
    icon: '⚡',
    description: "Google's mobile-optimised compound-scaled CNN.",
  },
  {
    id: 'convnext',
    name: 'ConvNeXt-Tiny',
    shortName: 'ConvNeXt',
    accuracy: '~87.5%',
    icon: '🔷',
    description: 'Modern CNN with transformer-style design choices.',
  },
  {
    id: 'yolo',
    name: 'YOLOv11 Nano',
    shortName: 'YOLO',
    accuracy: '—',
    icon: '🎯',
    description: 'Ultralytics YOLO nano — fastest inference.',
  },
]

export const DISEASE_INFO = {
  Blight_fungus: {
    displayName: 'Blight (Fungal)',
    emoji: '🍂',
    severity: 'High',
    color: '#dc2626',
    description: 'Fungal disease causing rapid death of plant tissue.',
    treatment: 'Apply copper-based fungicides. Remove infected tissue.',
  },
  Mosiac_Virus: {
    displayName: 'Mosaic Virus',
    emoji: '🟡',
    severity: 'Medium',
    color: '#d97706',
    description: 'Viral infection causing mottled yellow-green leaf patterns.',
    treatment: 'Remove infected plants. Control aphid vectors.',
  },
  Spider_Mite: {
    displayName: 'Spider Mite',
    emoji: '🕷️',
    severity: 'Medium',
    color: '#d97706',
    description: 'Tiny arachnid pests causing yellowing and leaf stippling.',
    treatment: 'Apply miticides or neem oil. Increase humidity.',
  },
  Thrip_pest: {
    displayName: 'Thrip Pest',
    emoji: '🪲',
    severity: 'Medium',
    color: '#d97706',
    description: 'Tiny insects piercing plant cells, causing silvery streaks.',
    treatment: 'Use insecticidal soap or spinosad. Apply sticky traps.',
  },
}
