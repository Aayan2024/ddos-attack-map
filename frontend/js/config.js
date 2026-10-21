export const CONFIG = Object.freeze({
    API_BASE: window.location.origin,

    AUTO_REFRESH_MS: 5 * 60 * 1000,
    REQUEST_TIMEOUT_MS: 120000,

    COLLECTION_LIMIT: 20,
    HISTORY_DAYS: 7,
    HISTORY_LIMIT: 5000,

    THREE_URL:
        'https://cdnjs.cloudflare.com/ajax/libs/' +
        'three.js/r128/three.min.js',

    ORBIT_CONTROLS_URL:
        'https://cdn.jsdelivr.net/npm/three@0.128.0/' +
        'examples/js/controls/OrbitControls.js',

    
    EARTH_TEXTURE_URL:
        'https://cdn.jsdelivr.net/npm/three-globe@2.18.8/' +
        'example/img/earth-blue-marble.jpg',

    SAMPLE_IPS: Object.freeze([
        '8.8.8.8',
        '1.1.1.1',
        '77.88.8.8',
        '180.76.76.76',
        '208.67.222.222'
    ])
});
