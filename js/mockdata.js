/**
 * mockdata.js
 * Static placeholder data used ONLY when the Flask API (started from the
 * backend notebook) cannot be reached. This lets a reviewer explore the full
 * UI immediately, and every real number is replaced the instant the notebook
 * backend responds. Clearly not a source of truth for medical claims.
 */
const MOCK = {
  overview: {
    total_patients: 20,
    predictions_today: 4,
    avg_confidence: 0.838,
    active_hospitals: 4,
  },
  recentPredictions: [
    { id:"s0",  patient:"Arjun Sharma",   age:54, finding:"Pneumonia",          confidence:0.91, hospital:"Hospital A", date:"2026-08-12", time:"09:41 AM" },
    { id:"s1",  patient:"Priya Nair",     age:38, finding:"Effusion",           confidence:0.86, hospital:"Hospital B", date:"2026-08-12", time:"10:05 AM" },
    { id:"s2",  patient:"Ramesh Gupta",   age:67, finding:"Cardiomegaly",       confidence:0.79, hospital:"Hospital C", date:"2026-08-12", time:"10:22 AM" },
    { id:"s3",  patient:"Sunita Patel",   age:45, finding:"Atelectasis",        confidence:0.83, hospital:"Hospital D", date:"2026-08-12", time:"10:47 AM" },
    { id:"s4",  patient:"Vikram Reddy",   age:59, finding:"No Finding",         confidence:0.95, hospital:"Hospital A", date:"2026-08-11", time:"08:30 AM" },
  ],
  history: [
    { id:"s0",  patient:"Arjun Sharma",   age:54, finding:"Pneumonia",          confidence:0.91, hospital:"Hospital A", date:"2026-08-12", time:"09:41 AM", gradcam_url:null },
    { id:"s1",  patient:"Priya Nair",     age:38, finding:"Effusion",           confidence:0.86, hospital:"Hospital B", date:"2026-08-12", time:"10:05 AM", gradcam_url:null },
    { id:"s2",  patient:"Ramesh Gupta",   age:67, finding:"Cardiomegaly",       confidence:0.79, hospital:"Hospital C", date:"2026-08-12", time:"10:22 AM", gradcam_url:null },
    { id:"s3",  patient:"Sunita Patel",   age:45, finding:"Atelectasis",        confidence:0.83, hospital:"Hospital D", date:"2026-08-12", time:"10:47 AM", gradcam_url:null },
    { id:"s4",  patient:"Vikram Reddy",   age:59, finding:"No Finding",         confidence:0.95, hospital:"Hospital A", date:"2026-08-11", time:"08:30 AM", gradcam_url:null },
    { id:"s5",  patient:"Deepa Menon",    age:42, finding:"Infiltration",       confidence:0.88, hospital:"Hospital B", date:"2026-08-11", time:"09:15 AM", gradcam_url:null },
    { id:"s6",  patient:"Karan Singh",    age:71, finding:"Pneumothorax",       confidence:0.76, hospital:"Hospital C", date:"2026-08-11", time:"11:02 AM", gradcam_url:null },
    { id:"s7",  patient:"Meera Iyer",     age:33, finding:"No Finding",         confidence:0.97, hospital:"Hospital D", date:"2026-08-11", time:"02:18 PM", gradcam_url:null },
    { id:"s8",  patient:"Ananya Das",     age:50, finding:"Edema",              confidence:0.81, hospital:"Hospital A", date:"2026-08-10", time:"08:55 AM", gradcam_url:null },
    { id:"s9",  patient:"Rohit Verma",    age:62, finding:"Consolidation",      confidence:0.84, hospital:"Hospital B", date:"2026-08-10", time:"09:40 AM", gradcam_url:null },
    { id:"s10", patient:"Fatima Khan",    age:48, finding:"Mass",               confidence:0.72, hospital:"Hospital C", date:"2026-08-10", time:"11:30 AM", gradcam_url:null },
    { id:"s11", patient:"Siddharth Rao",  age:55, finding:"Pleural_Thickening", confidence:0.78, hospital:"Hospital D", date:"2026-08-10", time:"01:20 PM", gradcam_url:null },
    { id:"s12", patient:"Pooja Mishra",   age:29, finding:"No Finding",         confidence:0.93, hospital:"Hospital A", date:"2026-08-09", time:"10:10 AM", gradcam_url:null },
    { id:"s13", patient:"Aditya Kumar",   age:73, finding:"Emphysema",          confidence:0.80, hospital:"Hospital B", date:"2026-08-09", time:"11:45 AM", gradcam_url:null },
    { id:"s14", patient:"Nisha Chopra",   age:41, finding:"Atelectasis",        confidence:0.87, hospital:"Hospital C", date:"2026-08-09", time:"02:00 PM", gradcam_url:null },
    { id:"s15", patient:"Manish Joshi",   age:66, finding:"Effusion",           confidence:0.89, hospital:"Hospital D", date:"2026-08-08", time:"09:05 AM", gradcam_url:null },
    { id:"s16", patient:"Kavita Pillai",  age:37, finding:"Fibrosis",           confidence:0.74, hospital:"Hospital A", date:"2026-08-08", time:"10:35 AM", gradcam_url:null },
    { id:"s17", patient:"Suresh Bhat",    age:58, finding:"Cardiomegaly",       confidence:0.82, hospital:"Hospital B", date:"2026-08-08", time:"12:00 PM", gradcam_url:null },
    { id:"s18", patient:"Anjali Tiwari",  age:44, finding:"Infiltration",       confidence:0.85, hospital:"Hospital C", date:"2026-08-07", time:"08:20 AM", gradcam_url:null },
    { id:"s19", patient:"Rajesh Nambiar", age:69, finding:"Nodule",             confidence:0.77, hospital:"Hospital D", date:"2026-08-07", time:"03:15 PM", gradcam_url:null },
  ],
  diseaseStats: {
    labels: ["No Finding", "Infiltration", "Effusion", "Atelectasis", "Pneumonia", "Cardiomegaly", "Pneumothorax", "Mass"],
    counts: [3, 2, 2, 2, 1, 2, 1, 1],
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
    { title: "Disease count plot", desc: "Frequency of each label across the full dataset.", url: "assets/figures/disease_distribution_bar.png" },
    { title: "Age vs. finding boxplot", desc: "Spread of patient age within each diagnosis.", url: "assets/figures/age_vs_finding_boxplot.png" },
    { title: "Sample X-ray grid", desc: "Representative images per class, pre-augmentation.", url: "assets/figures/sample_xray_grid.png" },
    { title: "Pixel intensity histogram", desc: "Normalized pixel value distribution after preprocessing.", url: "assets/figures/pixel_intensity_histogram.png" },
    { title: "Class imbalance chart", desc: "Ratio of minority to majority classes pre/post augmentation.", url: "assets/figures/class_imbalance_chart.png" },
    { title: "Hospital split pie chart", desc: "Share of the dataset allocated to each virtual hospital.", url: "assets/figures/hospital_split_pie.png" },
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
