import http from 'k6/http';
import { check, sleep } from 'k6';

// -----------------------------------------------------------------------------
// Load Testing Script for Cross-Environment Music Genre Inference API
// Conforms to Enterprise Performance Criteria: p95 < 1500ms, Error Rate < 1%
// -----------------------------------------------------------------------------

export const options = {
    vus: 10,              // 10 concurrent simulated users
    duration: '2m',       // 2 minutes sustained execution
    thresholds: {
        http_req_duration: ['p(95)<1500'], // 95% of queries completed within 1.5s
        http_req_failed: ['rate<0.01'],    // Technical failure rate strictly below 1%
    },
};

const sampleAudio = open('./data/audio/rock/rock_0001.wav', 'b');

export default function () {
    // 1. Healthcheck Request
    const healthRes = http.get('http://localhost:8000/health');
    check(healthRes, {
        'health status is 200': (r) => r.status === 200,
        'model_version present': (r) => r.json('model_version') !== undefined,
    });

    // 2. Real-time Inference Request
    const payload = {
        file: http.file(sampleAudio, 'test_song.wav', 'audio/wav'),
    };

    const res = http.post('http://localhost:8000/predict', payload);

    check(res, {
        'inference status is 200': (r) => r.status === 200,
        'has predicted_genre': (r) => r.json('predicted_genre') !== undefined,
        'has confidence': (r) => r.json('confidence') !== undefined,
        'has ai_status': (r) => ['OK', 'UNCERTAIN', 'FALLBACK'].includes(r.json('status')),
    });

    sleep(1); // 1 second think time between uploads
}
