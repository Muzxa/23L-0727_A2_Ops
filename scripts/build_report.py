"""Build the Tasks 1-2 PDF from actual exported Colab evidence."""
import json
import csv
import subprocess
from pathlib import Path
from html import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[1]
E = ROOT/'outputs/23L-0727_A2_Ops'
if not E.exists():
    E = ROOT/'submission/evidence'
C = E/'colab'
T = C/'training_cached'
D = E/'data_evidence'
S = E/'screenshots'
OUT = ROOT/'submission/23L-0727_VisionGuard.pdf'
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyVG',fontName='Helvetica',fontSize=9.5,leading=13,spaceAfter=7,textColor=colors.HexColor('#26354a')))
styles.add(ParagraphStyle(name='CaptionVG',fontSize=8,leading=10,spaceBefore=4,spaceAfter=10,textColor=colors.HexColor('#546378')))
styles.add(ParagraphStyle(name='TitleVG',fontName='Helvetica-Bold',fontSize=30,leading=35,spaceAfter=18,textColor=colors.HexColor('#122d4d')))
styles.add(ParagraphStyle(name='HVG',fontName='Helvetica-Bold',fontSize=17,leading=21,spaceBefore=5,spaceAfter=12,textColor=colors.HexColor('#122d4d')))
styles.add(ParagraphStyle(name='SmallVG',fontName='Helvetica',fontSize=7.5,leading=10,spaceAfter=5))
story=[]
fig=0

def js(p): return json.loads(Path(p).read_text())
def rows(p):
    with open(p,newline='') as f: return list(csv.DictReader(f))
def para(text,style='BodyVG'): story.append(Paragraph(text,styles[style]))
def heading(text): para(text,'HVG')
def page(text):
    if story: story.append(PageBreak())
    heading(text)
def table(data,widths=None,small=False):
    sty=styles['SmallVG' if small else 'BodyVG']
    wrapped=[[Paragraph(escape(str(v)).replace('\n','<br/>'),sty) for v in row] for row in data]
    tb=Table(wrapped,colWidths=widths,repeatRows=1,hAlign='LEFT')
    tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5edf6')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#9aadc3')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dbe2ea')),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.extend([tb,Spacer(1,8)])
def image(path,caption,maxh=270):
    global fig
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(path)
    iw,ih=ImageReader(str(path)).getSize()
    scale=min(493/iw,maxh/ih)
    fig+=1
    story.append(KeepTogether([Image(str(path),width=iw*scale,height=ih*scale,hAlign='CENTER'),Paragraph(f'Figure {fig}. '+caption,styles['CaptionVG'])]))
def shot(name,caption,maxh=360): image(S/name,caption,maxh)
def footer(canvas,doc):
    canvas.saveState()
    w,h=doc.pagesize
    canvas.setStrokeColor(colors.HexColor('#b7c8dc'));canvas.line(48,42,w-48,42)
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#546378'))
    canvas.drawString(48,29,'23L-0727_A2_Ops  |  VisionGuard  |  Tasks 1 and 2')
    canvas.drawRightString(w-48,29,str(doc.page));canvas.restoreState()

def build():
    assert (T/'COMPLETE.json').exists(), 'Full real-data run is required before report generation'
    selection=js(T/'selection_before_test.json')
    metrics={m:js(T/f'evaluation/{m}/metrics.json') for m in ['baseline','transfer']}
    env=js(T/'environment.json')
    links=js(E/'submission_metadata.json')
    para('VISIONGUARD','TitleVG')
    para('AI-generated vs real image classification','HVG')
    para('Task 1: Data Understanding and Versioning<br/>Task 2: Model Development with TensorFlow/Keras')
    story.append(Spacer(1,24))
    table([['Submission detail','Value'],['Name',links.get('name','Hayyan Haider')],['Roll number','23L-0727'],['Project','23L-0727_A2_Ops'],['Date','6 November 2026 (Asia/Karachi)'],['Graded source commit',links['commit']],['GitHub',links['github']],['DagsHub',links['dagshub']],['Colab notebook','https://colab.research.google.com/drive/1DUeUYKEAaNH7_sl6O8K48r6MESux_Htl'],['Scope','Tasks 1 and 2 only. Deployment/API work belongs to later tasks.']],[116,377])
    para('<b>AI assistance disclosure.</b> OpenAI Codex helped prepare and repair the audit and training code, operated the Colab experiment, inspected evidence, and drafted this report. The reported measurements come from the actual assignment dataset and saved experiment outputs. The student should review the code, results, and interpretations before submission.')
    para('The experiment uses folder labels (AI=1, real=0). These labels are dataset annotations; a model score is not proof of image authenticity. All test figures reuse one inference pass per final model.')

    page('Task 1 - Data Understanding and Versioning')
    heading('1.1 - Environment and GPU')
    para('The local project uses an isolated Python virtual environment, separating its dependencies from system Python. The graded model experiment ran in a separate Google Colab runtime with an NVIDIA T4 GPU. TensorFlow GPU detection and the runtime/package record are shown below; exact installed package versions are supplied in requirements-lock.txt.')
    table([['Setting','Observed value'],['Python',env['python'].split('\n')[0]],['TensorFlow / Keras',env['tensorflow']+' / '+env['keras']],['GPU devices',', '.join(env['gpu'])],['CUDA / cuDNN',str(env['build'].get('cuda_version'))+' / '+str(env['build'].get('cudnn_version'))],['Random seed / image size / batch','42 / 224 x 224 RGB / 32']],[135,358])
    shot('environment.png','Actual Colab package versions and GPU evidence, with project context.',312)
    para('<b>Commands and choices.</b> python -m venv creates isolation; pip installs the declared dependencies. TensorFlow list_physical_devices checks GPU visibility; pip freeze records exact versions. Keeping Colab\'s supplied TensorFlow/CUDA combination avoids replacing its working GPU stack.')
    para('<b>Observed issue.</b> Google Drive was full, preventing notebook saves. Training checkpoints and outputs were therefore written under /content/23L-0727_evidence and downloaded to the local submission bundle. Runtime-local files alone are not persistent backups.')

    page('1.2 - Original dataset audit')
    para('The script recursively walks the original Kaggle files. For each candidate image it records class, file size, width, height, colour mode, format, byte SHA-256, decoded RGB pixel SHA-256, and any decoding error. Pillow verify plus a full decode checks readability. The nearest recognized parent folder assigns the label: AiArtData is AI and RealArt is real.')
    table([['Class','Files','Readable','Width min/median/max','Height min/median/max'],['AI','538','538','225 / 1028.5 / 6000','135 / 800 / 4001'],['Real','435','435','183 / 940 / 8495','158 / 720 / 7091']],[42,40,53,179,179],True)
    table([['Class','File bytes: minimum','Median','Maximum'],['AI','11,738','199,802.5','5,626,183'],['Real','11,311','144,982','10,195,352']],[60,145,144,144])
    para('There are 973 readable images and no unreadable candidates. One duplicate group contributes one extra file; decoded-pixel deduplication also removes one extra image. No cross-label pixel conflicts were found. AI images account for 55.3% of the original collection, so accuracy should be considered alongside precision, recall, F1, and ROC-AUC.')
    table([['Split','Real','AI','Total'],['Train','304','376','680'],['Validation','65','81','146'],['Test','66','80','146']],[123,123,123,124])
    para('Deduplication happens before a seeded, stratified 70/15/15 split. Split lists retain original paths and hashes, making membership auditable. The retained pixel-hash sets are disjoint across train/validation/test. This only detects identical images: near duplicates and shared web sources may still leak visual information.')
    shot('audit.png','Executed audit and split evidence for the assignment dataset.',270)

    page('1.3 - Visual EDA: sample grid and widths')
    image(D/'samples.png','Deterministic original-image samples from each class (seed 42).',235)
    para(links['sample_interpretation'])
    image(D/'width_histogram.png','Original image width distribution by class; horizontal axis is logarithmic.',220)
    para('Median width is 1028.5 pixels for AI images and 940 pixels for real images, with substantial overlap. Real images have the more extreme maximum (8495 versus 6000 pixels). Resolution is therefore a possible source shortcut and compute cost; resizing makes tensor shapes consistent but does not establish that the detector has learned generation artifacts.')

    page('1.3 - Visual EDA: heights and file sizes')
    image(D/'height_histogram.png','Original image heights by class on a logarithmic axis.',230)
    para('Median height is 800 pixels for AI images and 720 for real images. Both distributions span small and large images, and real images reach 7091 pixels compared with 4001 for AI. A fixed 224-pixel input reduces memory use, while potentially discarding fine forensic details.')
    image(D/'size_bytes_histogram.png','Original file-size distributions by class on a logarithmic axis.',230)
    para('The AI median file size is about 199.8 kB, compared with 145.0 kB for real images, although the largest real file is about 10.2 MB. These are original compressed file sizes, not processed PNG sizes or tensor memory. Compression format and source quality can correlate with class, so file size alone would be an unreliable authenticity signal.')

    page('1.4 - Git and DVC: toy round trip')
    para('A new empty directory was initialized with Git and DVC. tiny.csv initially contained two labeled rows; dvc add created its pointer, and Git committed the metadata. A third row was added and committed as a second version. Both data versions were pushed to a configured local-folder DVC remote, which the assignment permits.')
    toy=js(E/'dvc_toy/result.json')
    table([['Toy evidence','Observed value'],['Version 1 commit',toy['first_commit']],['Version 2 commit',toy['second_commit']],['Restored version',toy['restored_commit']],['Recovery assertion',str(toy['verified'])]],[112,381],True)
    para('git checkout selected the first commit and dvc checkout restored its two-row content. The toy CSV and its local DVC cache were then removed. dvc pull recovered the older bytes from the remote, and the script asserted exact content equality. Deleting the cache makes this stronger than recovering from a local cached copy.')
    shot('dvc_toy.png','Both toy Git commits published on the DagsHub dvc-toy branch. Recovery is verified in the accompanying transcript and result.json.',350)
    para('git init starts metadata history; dvc init creates DVC configuration; dvc add hashes data and writes pointers; dvc remote add chooses object storage; dvc push uploads objects; dvc pull retrieves the objects named by the checked-out pointer. The two toy commits are also published to the DagsHub project as described in the accompanying repository evidence.')

    page('1.4 - Apply DVC to the real dataset')
    para('The original dataset, normalized RGB copies, and split manifests were tracked separately with DVC. dvc push uploaded their objects to the persistent local remote under versioned_storage/dvc_remote. Git stores the small pointers and configuration; a bare Git backup preserves metadata independently of the working tree.')
    pointer=(ROOT/'data/raw.dvc').read_text()
    para('<b>Actual raw pointer:</b><br/>'+escape(pointer).replace('\n','<br/>'),'SmallVG')
    para('The pointer contains the content checksum, byte size, file count, hash algorithm, and path. For a directory, its .dir object identifies the file manifest. The audit uses SHA-256 for source identity; DVC\'s content checksum serves its own storage protocol. Large images remain outside Git because binary versions inflate history and do not produce useful text diffs.')
    para('The raw DVC pointer counts 975 files, while the image audit counts 973 candidates: DVC also includes two hidden .DS_Store files ignored by the image audit. This difference does not indicate two unreadable images.')
    shot('dvc_dataset.png','Browser view of the actual saved raw pointer and freshly executed DVC remote status. The log file is included in the evidence bundle.',300)
    page('1.4 - Published project and version history')
    para('Screenshots are cropped browser captures. The DagsHub excerpts omit the browser address bar and account header; the displayed repository records and results are unchanged. The software cursor overlay is excluded. The GitHub project is mirrored to DagsHub under the required roll-number project name. The dvc-toy branch preserves both CSV versions, while main contains the audited dataset pointers and model development source. Figure '+str(fig+1)+' shows the published project; large data objects remain in the permitted local DVC remote.')
    shot('dagshub.png','DagsHub repository excerpt: branch, source commit, and tracked files; account header cropped.',355)
    para('The local remote satisfies the permitted storage option. Reproducing on another computer requires access to that remote or an exported copy and an updated remote path; a Git clone alone cannot reconstruct excluded image bytes.')

    page('Task 2 - Model Development with TensorFlow/Keras')
    heading('2.1 - Input pipeline and augmentation')
    para('Saved CSV split lists supply file paths and float labels. tf.data reads each RGB PNG, decodes three channels, resizes with bilinear interpolation and antialiasing, and pads to 224 x 224 without stretching. This matches MobileNetV2\'s standard input and limits GPU memory, at the cost of losing small artifacts. Padding can itself become a shortcut if aspect ratios correlate with labels.')
    para('The deterministic resized tensors are cached in memory before training shuffle. Training reshuffles each epoch with seed 42; batches contain up to 32 examples, and one batch is prefetched. Decode parallelism is capped at two to limit memory. Cache avoids repeatedly decoding large PNGs, and does not cache augmentation. Validation/test preserve manifest order and use no augmentation.')
    para('Training-only augmentation lives inside each model: horizontal flips, rotation up to 0.025 turns, zoom factor 0.08, and contrast factor 0.08. Reflect padding reduces blank fill during geometric augmentation. Keras disables these random transforms during evaluation; saved preprocessing therefore travels with the model.')
    image(T/'augmented_batch.png','A genuine augmented training batch. Labels come from the manifest.',355)
    para('<b>Observed repair.</b> One processed PNG failed TensorFlow decoding. Its source byte hash and decoded pixel hash were checked against the original audit, then its RGB pixels were re-encoded with TensorFlow. All 972 split images passed decoding and pixel-hash verification; no examples or labels were dropped.')

    page('2.2 - Scratch CNN architecture')
    para('The scratch network has three convolution blocks (32, 64, 128 filters). Each 3 x 3 convolution shares local feature weights; BatchNorm stabilizes activations, ReLU adds nonlinearity, and max pooling reduces spatial size and computation. Global average pooling avoids a very large flattened classifier. A 64-unit ReLU layer combines channels; dropout 0.4 regularizes, and one sigmoid produces an AI score.')
    layers=rows(T/'baseline/layers.csv')
    table([['Layer','Output shape','Parameters']]+[[r['layer'],r['output_shape'],f"{int(r['parameters']):,}"] for r in layers if '/' not in r['layer']],[214,179,100],True)
    para('The loss is binary cross entropy because the target is binary and the output is a sigmoid. Adam starts at 0.001, providing adaptive updates. Training monitors binary accuracy, AUC, precision, and recall; the latter classification metrics use 0.5 during fit. Final metrics instead use thresholds chosen solely from validation. Sigmoid scores are not guaranteed calibrated authenticity probabilities.')
    page('2.2 - Model summary evidence')
    para('Figure '+str(fig+1)+' shows the actual layer evidence exported from the trained scratch model. Shapes follow the data flow in the preceding table: three pooling operations reduce 224 x 224 to 28 x 28, and global average pooling reduces each feature map to one channel value. The screenshot focuses on layers with parameters. Convolutional layers dominate this small network; augmentation and pooling have no learned parameters.')
    shot('model_summary.png','Executed weighted-layer excerpt of the scratch CNN, with output shapes and parameter counts.',380)

    page('2.3 - Transfer learning in two stages')
    para('MobileNetV2 uses ImageNet pretrained weights with include_top=False. The added classifier consists of global average pooling, dropout 0.3, and one sigmoid. Embedded rescaling maps 0-255 pixels to [-1,1], matching MobileNetV2 preprocessing. ImageNet features provide a useful initialization, while the authenticity task differs from the original object classification task.')
    para('Stage 1 freezes the backbone and trains the head with Adam at 0.001. Stage 2 unfreezes only the last block\'s weighted non-BatchNorm layers and the terminal convolution, then recompiles with learning rate 0.00001 (100 times smaller). BatchNorm remains frozen and the backbone is called in inference mode to preserve its moving statistics.')
    for stage in ['head','finetune']:
        image(T/stage/'curves.png',f'{stage}: loss and accuracy curves, with minimum validation-loss epoch marked.',190)
        names=js(T/stage/'trainable_layers.json')
        para('<b>'+stage+' trainable weighted layers:</b> '+', '.join(names),'SmallVG')
    para('The best validation loss fell from 0.53795 for the frozen head to 0.49179 after last-block adaptation. This supports useful adaptation on this validation split, while the later tuner winner achieved 0.47383. The frozen-head best epoch was the budget limit (10), so early stopping did not establish convergence for that stage.')

    page('2.4 - Training discipline and checkpoints')
    para('Python, NumPy, and TensorFlow seeds are fixed to 42, and TensorFlow deterministic operations are enabled. Exact reproducibility still depends on hardware, CUDA, and package versions. A checkpoint is saved only when validation loss improves. Early stopping uses patience 4 and restores best weights; ReduceLROnPlateau halves the learning rate after 2 stagnant epochs with minimum 1e-7.')
    table([['Stage','Epoch budget','Epochs run','Best epoch','Best validation loss']]+[[stage,budget,js(T/stage/'best_epoch.json')['epochs_run'],js(T/stage/'best_epoch.json')['best_epoch'],f"{js(T/stage/'best_epoch.json')['best_val_loss']:.5f}"] for stage,budget in [('baseline',20),('head',10),('finetune',15)]],[90,85,85,85,148])
    image(T/'baseline/curves.png','Scratch CNN training curves, with the selected validation-loss checkpoint marked.',205)
    para('Scratch training loss generally decreases, while validation loss improves only to epoch 6 and worsens later. Training accuracy rises above validation accuracy, which stays near the class-majority level. This gap is consistent with overfitting or dataset mismatch; the curves alone cannot identify its cause.')
    para('The transfer candidates (frozen head, last block, and tuner winner) were compared by validation loss before test access. The selected candidate was <b>'+selection['transfer_candidate']+'</b>. Thresholds and split hashes were saved in selection_before_test.json, then an exclusive test-start marker prevented a second final evaluation in this run directory.')
    shot('training_complete.png','Actual completed experiment evidence and stage results.',240)

    page('2.5 - Held-out evaluation and decision threshold')
    para('AI is the positive class. A real photograph incorrectly called fake can damage trust, so the predeclared validation objective is 2 x false positives + false negatives. This 2:1 ratio is a product judgment, not a measured business cost. The algorithm checks every attainable validation boundary and chooses the higher threshold when costs tie. Test labels never influence this selection.')
    table([['Model','Threshold','Accuracy','Precision','Recall','F1','ROC-AUC']]+[[m,f"{v['threshold']:.5f}"]+[f"{v[k]:.4f}" for k in ['accuracy','precision','recall','f1','roc_auc']] for m,v in metrics.items()],[69,75,69,70,69,70,71],True)
    table([['Model','True real (TN)','Real called AI (FP)','AI missed (FN)','True AI (TP)']]+[[m,v['tn'],v['fp'],v['fn'],v['tp']] for m,v in metrics.items()],[80,98,119,98,98],True)
    for m,v in metrics.items():
        para(f"<b>{m}:</b> {v['fp']} of 66 real images were called AI, and {v['fn']} of 80 AI images were missed. Precision measures correctness of AI accusations; recall measures the fraction of AI images caught. ROC-AUC uses continuous scores and does not depend on the selected threshold.")
    para(links['evaluation_interpretation'])
    para('The 146-image test set is small and comes from the same collection as training. These are point estimates without a new-source or new-generator guarantee. Source artifacts, label uncertainty, compression, and unseen generators limit generalization; the detector should support review rather than establish authenticity.')
    page('2.5 - Executed metric evidence')
    para('Figure '+str(fig+1)+' records the final saved test results in the actual Colab notebook. The report tables use the unrounded JSON values, whereas this compact display rounds to four decimal places. Confusion-matrix counts preserve the exact errors behind precision and recall; all values refer to the same 146 held-out images.')
    shot('evaluation.png','Executed test metric comparison with the project context.',355)

    page('2.5 - Confusion matrices and ROC curves')
    for m in ['baseline','transfer']:
        a,b=T/f'evaluation/{m}/confusion_matrix.png',T/f'evaluation/{m}/roc.png'
        cells=[]
        for p in [a,b]:
            iw,ih=ImageReader(str(p)).getSize();cells.append(Image(str(p),width=240,height=240*ih/iw))
        story.append(Table([cells],colWidths=[246,247]))
        para(m+': confusion matrix (rows=true, columns=prediction) and held-out ROC curve.','CaptionVG')
    para('Each final model predicts the test set exactly once. Probabilities, row-level predictions, and labels are saved, and the confusion matrix, ROC, metrics, and error grids are derived from that same pass. Accuracy is (TP+TN)/N; precision=TP/(TP+FP); recall=TP/(TP+FN); F1 is their harmonic mean. No model was retrained after the held-out metrics were exposed.')

    for m in ['transfer']:
        page('2.6 - Error analysis: '+m)
        image(T/f'evaluation/{m}/misclassified.png',f'First ten available {m} errors in manifest order; true and predicted labels and AI scores are shown.',235)
        errs=rows(T/f'evaluation/{m}/error_analysis.csv')
        assert all(r['visual_observation'] and r['hypothesis_to_verify'] for r in errs)
        table([['Image','Visible observation','Hypothesis and check']]+[[i+1,r['visual_observation'],r['hypothesis_to_verify']] for i,r in enumerate(errs)],[36,204,253],True)
        para('Images are numbered left-to-right, top-to-bottom. Sorted paths place AI images first, so these first ten available errors are all false negatives; the confusion matrices cover both error types. These are visual observations and testable hypotheses, not proven causes. Proposed diagnostics were not used to retune the model or threshold.')

    page('2.7 - Keras Tuner: Fashion-MNIST toy')
    para('The toy uses a tiny flattened Fashion-MNIST classifier with one hidden ReLU layer and a 10-class softmax. A fixed stratified subset of 10,000 training and 2,000 validation images is taken from the built-in training partition; its test partition remains unused. Pixels are scaled to [0,1], with sparse categorical cross entropy for integer class labels.')
    para('Random search explores 32/64/128 hidden units and Adam learning rates 0.01/0.001/0.0001. Eight trials run up to five epochs with early stopping (patience 2). The objective is minimum validation loss, so lower is better.')
    toyrows=rows(T/'tuner_toy/trials.csv')
    table([['Trial','Status','Units','Learning rate','Validation loss']]+[[r['trial_id'],r['status'],r['units'],r['learning_rate'],f"{float(r['val_loss']):.5f}"] for r in toyrows],[55,100,70,115,153])
    best=js(T/'tuner_toy/best_hyperparameters.json')
    para('Best toy settings: <b>'+escape(str(best))+'</b>. This lab demonstrates tuning mechanics on an unrelated built-in dataset; its score is not VisionGuard performance.')
    shot('tuner_toy.png','Actual Fashion-MNIST toy search results.',300)

    page('2.7 - Applied VisionGuard tuning')
    para('Every trial starts from the same trained frozen-head checkpoint. Random search varies learning rate {1e-5, 3e-5, 1e-4}, head dropout {0.2, 0.4, 0.5}, and {1, 2, 4} unfrozen weighted layers within the final MobileNetV2 block/terminal convolution. Earlier layers and BatchNorm remain frozen. Each of eight trials runs up to five epochs with early stopping and learning-rate reduction.')
    trials=rows(T/'tuner/trials.csv')
    assert sum(r['status']=='COMPLETED' for r in trials)>=8
    table([['Trial','Status','LR','Dropout','Unfrozen','Val loss']]+[[r['trial_id'],r['status'],r['learning_rate'],r['dropout'],r['unfrozen_layers'],f"{float(r['val_loss']):.5f}"] for r in trials],[40,100,80,75,75,123],True)
    best=js(T/'tuner/best_hyperparameters.json')
    para('Best applied settings: <b>'+escape(str(best))+'</b>. The actual winning checkpoint is retained; the final transfer candidate is selected against the head and standalone fine-tune candidates using validation loss.')
    page('2.7 - Applied tuner evidence and interpretation')
    para('Figure '+str(fig+1)+' confirms that all eight applied random-search trials completed and records the winning settings. Trial IDs are zero-based, so winning trial 4 is the fifth trial. These are validation results; test data did not choose the hyperparameters.')
    shot('tuner_applied.png','Eight completed applied trials and winning settings.',310)
    ranges=js(T/'tuner/descriptive_effect_ranges.json')
    winner=max(ranges,key=ranges.get)
    table([['Parameter','Range of marginal mean validation losses']]+[[k,f'{v:.6f}'] for k,v in ranges.items()],[180,313])
    para('<b>'+winner+'</b> has the widest observed marginal mean-loss range, and thus the strongest exploratory association in these trials. This is not causal importance: multiple settings change together, each value has few observations, and optimization is stochastic. A controlled one-factor comparison would be needed to isolate effects.')
    para('The marginal mean validation loss was 0.52172 at 1e-5 (2 trials), 0.51140 at 3e-5 (3), and 0.48805 at 1e-4 (3). Larger updates may have adapted the pretrained final layers faster within the short five-epoch budget. This is a hypothesis about the observed association, not proof that 1e-4 is universally best.')
    para('Choosing the minimum of many validation scores can select random favorable noise and overfit the validation split. The test set is consequently opened only after candidate and threshold selection. Further model changes after this report would require a fresh held-out set for an unbiased final check.')

    page('Reproduction, troubleshooting, and references')
    para('The source repository contains the audit, pipeline/models, tuning, threshold/evaluation code, and standalone notebook. The evidence bundle includes the exact environment, split manifests, source hashes, curves, checkpoints, tuner state, and saved predictions. The report builder reads these outputs and refuses to build without COMPLETE.json and at least eight completed applied trials.')
    table([['Observed issue','Resolution and evidence'],['PNG decoding failure','Lossless re-encoding from the audited original; 972/972 pixel hashes verified, one file repaired.'],['Slow repeated full-resolution decode','Cache resized deterministic inputs before per-epoch shuffle; augmentation remains dynamic.'],['Google Drive storage quota','Use runtime-local output storage and download evidence/checkpoints. Existing Drive content was not deleted.']],[140,353])
    para('For a new run: create a virtual environment; install requirements; audit the originals; verify labels and split hashes; run the PNG integrity repair; execute scripts.run_training with a new output directory. Use the provided unchanged settings for a pre-test resume. After test access, read saved predictions rather than repeating model selection.')
    para('Task 3 experiment tracking and Tasks 4-8 serving, database, containers, cloud deployment, and API integration are outside this Tasks 1-2 submission. The optional Grad-CAM bonus is also outside this report. Local saved experiment evidence is not claimed as remote MLflow tracking.')
    for label,url in [('Assignment dataset','https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images'),('TensorFlow transfer learning','https://www.tensorflow.org/guide/keras/transfer_learning'),('Keras Tuner getting started','https://keras.io/keras_tuner/getting_started/'),('DVC remote storage','https://dvc.org/doc/user-guide/data-management/remote-storage')]:
        para(f'<b>{label}:</b> <link href="{url}" color="#1c5c91">{url}</link>','SmallVG')
    OUT.parent.mkdir(exist_ok=True)
    doc=SimpleDocTemplate(str(OUT),pagesize=(595.276,841.89),rightMargin=51,leftMargin=51,topMargin=47,bottomMargin=56,title='VisionGuard - Tasks 1 and 2 - 23L-0727',author=links.get('name','Hayyan Haider'))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(OUT)

if __name__=='__main__': build()
