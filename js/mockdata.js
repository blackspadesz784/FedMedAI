/**
 * mockdata.js
 * Static placeholder data used ONLY when the Flask API (started from the
 * backend notebook) cannot be reached. This lets a reviewer explore the full
 * UI immediately, and every real number is replaced the instant the notebook
 * backend responds. Clearly not a source of truth for medical claims.
 */
const MOCK = {
  overview: {
    total_patients: 1284,
    predictions_today: 37,
    avg_confidence: 0.911,
    active_hospitals: 4,
  },
  recentPredictions: [
    { patient: "R. Kapoor", finding: "Pneumonia", confidence: 0.94, hospital: "Hospital A", time: "9:41 AM" },
    { patient: "S. Iyer", finding: "No Finding", confidence: 0.88, hospital: "Hospital C", time: "9:22 AM" },
    { patient: "M. Chen", finding: "Cardiomegaly", confidence: 0.79, hospital: "Hospital B", time: "8:57 AM" },
    { patient: "A. Farah", finding: "Effusion", confidence: 0.91, hospital: "Hospital D", time: "8:30 AM" },
    { patient: "J. Okafor", finding: "Atelectasis", confidence: 0.83, hospital: "Hospital A", time: "8:12 AM" },
    { patient: "L. Novak", finding: "No Finding", confidence: 0.95, hospital: "Hospital B", time: "7:58 AM" },
  ],
  history: [
    { patient: "R. Kapoor", age: 54, finding: "Pneumonia", confidence: 0.94, hospital: "Hospital A", date: "2026-07-30" },
    { patient: "S. Iyer", age: 61, finding: "No Finding", confidence: 0.88, hospital: "Hospital C", date: "2026-07-30" },
    { patient: "M. Chen", age: 47, finding: "Cardiomegaly", confidence: 0.79, hospital: "Hospital B", date: "2026-07-29" },
    { patient: "A. Farah", age: 39, finding: "Effusion", confidence: 0.91, hospital: "Hospital D", date: "2026-07-29" },
    { patient: "J. Okafor", age: 66, finding: "Atelectasis", confidence: 0.83, hospital: "Hospital A", date: "2026-07-28" },
    { patient: "L. Novak", age: 29, finding: "No Finding", confidence: 0.95, hospital: "Hospital B", date: "2026-07-27" },
    { patient: "P. Singh", age: 72, finding: "Infiltration", confidence: 0.86, hospital: "Hospital C", date: "2026-07-26" },
    { patient: "T. Almeida", age: 58, finding: "Pneumonia", confidence: 0.77, hospital: "Hospital D", date: "2026-07-25" },
  ],
  diseaseStats: {
    labels: ["No Finding", "Infiltration", "Effusion", "Atelectasis", "Pneumonia", "Cardiomegaly", "Pneumothorax", "Mass"],
    counts: [412, 231, 198, 176, 154, 121, 88, 61],
  },
  ageHistogram: {
    labels: ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90"],
    counts: [12, 34, 58, 97, 156, 210, 240, 175, 66],
  },
  correlationLabels: ["Effusion", "Atelectasis", "Infiltration", "Pneumonia", "Cardiomegaly"],
  correlationMatrix: [
    [1.00, 0.42, 0.31, 0.18, 0.27],
    [0.42, 1.00, 0.38, 0.22, 0.15],
    [0.31, 0.38, 1.00, 0.46, 0.11],
    [0.18, 0.22, 0.46, 1.00, 0.09],
    [0.27, 0.15, 0.11, 0.09, 1.00],
  ],
  datasetStats: {
    total_images: 5606,
    train: 4030,
    val: 806,
    test: 770,
    classes: 8,
  },
  classBalance: {
    labels: ["No Finding", "Infiltration", "Effusion", "Atelectasis", "Pneumonia", "Cardiomegaly", "Pneumothorax", "Mass"],
    counts: [1780, 980, 845, 760, 640, 512, 380, 309],
  },
  vizGallery: [
    { title: "Disease count plot", desc: "Frequency of each label across the full dataset." },
    { title: "Age vs. finding boxplot", desc: "Spread of patient age within each diagnosis." },
    { title: "Sample X-ray grid", desc: "Representative images per class, pre-augmentation." },
    { title: "Pixel intensity histogram", desc: "Normalized pixel value distribution after preprocessing." },
    { title: "Class imbalance chart", desc: "Ratio of minority to majority classes pre/post augmentation." },
    { title: "Hospital split pie chart", desc: "Share of the dataset allocated to each virtual hospital." },
  ],
  flStatus: { current_round: 18, total_rounds: 25, global_auc: 0.93, status: "aggregating" },
  hospitals: [
    { name: "Hospital A", samples: 1401, epoch: "6/6", loss: 0.214, accuracy: 0.912, status: "Synced" },
    { name: "Hospital B", samples: 1387, epoch: "6/6", loss: 0.231, accuracy: 0.901, status: "Synced" },
    { name: "Hospital C", samples: 1412, epoch: "5/6", loss: 0.248, accuracy: 0.894, status: "Training" },
    { name: "Hospital D", samples: 1406, epoch: "6/6", loss: 0.226, accuracy: 0.905, status: "Synced" },
  ],
  flRounds: {
    labels: Array.from({ length: 18 }, (_, i) => `R${i + 1}`),
    accuracy: [0.61,0.67,0.71,0.75,0.78,0.80,0.82,0.84,0.855,0.865,0.875,0.885,0.89,0.897,0.902,0.907,0.911,0.915],
  },
  modelEval: {
    accuracy: 0.915, precision: 0.897, recall: 0.883, f1: 0.89, auc: 0.93, loss: 0.214,
    accLossCurve: {
      labels: Array.from({ length: 12 }, (_, i) => `Epoch ${i + 1}`),
      train_acc: [0.55,0.63,0.69,0.74,0.78,0.81,0.84,0.86,0.875,0.888,0.90,0.915],
      val_acc:   [0.52,0.60,0.66,0.71,0.75,0.78,0.80,0.82,0.835,0.845,0.855,0.86],
      train_loss:[0.98,0.81,0.68,0.58,0.50,0.44,0.39,0.35,0.32,0.30,0.28,0.26],
      val_loss:  [1.05,0.89,0.76,0.66,0.59,0.54,0.50,0.47,0.45,0.44,0.43,0.42],
    },
    roc: { fpr: [0,0.05,0.1,0.2,0.3,0.45,0.6,0.8,1], tpr: [0,0.38,0.55,0.7,0.8,0.88,0.93,0.97,1] },
    pr: { recall: [0,0.2,0.4,0.6,0.8,1], precision: [1,0.96,0.92,0.88,0.8,0.6] },
    confusion: {
      labels: ["No Finding", "Infiltration", "Effusion", "Pneumonia"],
      matrix: [
        [180, 8, 4, 2],
        [10, 140, 9, 5],
        [3, 11, 128, 6],
        [2, 4, 7, 96],
      ],
    },
  },
};
