function _subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function _cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function _normalize(a) {
  const length = Math.hypot(a[0], a[1], a[2]);
  return [a[0] / length, a[1] / length, a[2] / length];
}

export function computeCameraBasisVectors(eyePosition, targetPosition, upDirection) {
  const forwardAxis = _normalize(_subtract(eyePosition, targetPosition)); // eye - target
  const rightAxis = _normalize(_cross(upDirection, forwardAxis)); // cross(up, forward)
  const trueUpAxis = _cross(forwardAxis, rightAxis);
  return { rightAxis, trueUpAxis, forwardAxis };
}

export function computeLineSpanWidth({ eyeDistance, fovVertical, aspectRatio }) {
  return 2 * eyeDistance * Math.tan(fovVertical / 2) * aspectRatio;
}