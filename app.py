import streamlit as st
import numpy as np
import librosa, soundfile as sf
import pathlib, hashlib, subprocess, sys, io, csv, os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
st.set_page_config(page_title='Guitar Studio AI',page_icon='🎸',layout='wide')
st.markdown("""<style>
.stApp {background:#10151c;color:#edf4f5} [data-testid="stSidebar"] {background:#151d25} .stButton button,.stDownloadButton button {border-radius:9px;border:1px solid #32cdb5} [data-testid="stFileUploader"] {background:#18232c;border-radius:12px;padding:12px} .stTabs [data-baseweb="tab"] {font-size:17px;padding:12px 24px} </style>""",unsafe_allow_html=True)
st.sidebar.title('🎸 Guitar Studio AI')
st.sidebar.caption('ESTUDIO LOCAL · INTEL')
st.sidebar.markdown('**Separación · Acordes · Tablaturas**')
st.sidebar.info('Procesamiento local. Para 8 GB de RAM, comienza en modo Rápida.')
st.title('Guitar Studio AI')
st.caption('Estudio musical local · Separación, armonía y práctica')
root=pathlib.Path.home()/ 'Music'/'GuitarStudioAI'; root.mkdir(parents=True,exist_ok=True)
u=st.file_uploader('Abre una canción',type=['mp3','wav','m4a','flac','ogg'])
if not u: st.stop()
data=u.getvalue(); folder=root/hashlib.sha256(data).hexdigest()[:16]; folder.mkdir(exist_ok=True)
p=folder/('original'+pathlib.Path(u.name).suffix.lower()); p.write_bytes(data)
notes=['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']
@st.cache_data(show_spinner=False)
def chords(path):
 y,sr=librosa.load(path,sr=22050); h=librosa.effects.harmonic(y); c=librosa.feature.chroma_cqt(y=h,sr=sr); hop=512
 templates=[]; names=[]
 for r in range(12):
  for suffix,intervals in [('',[0,4,7]),('m',[0,3,7]),('7',[0,4,7,10]),('m7',[0,3,7,10]),('maj7',[0,4,7,11])]:
   t=np.zeros(12); t[(r+np.array(intervals))%12]=1; templates.append(t/np.linalg.norm(t)); names.append(notes[r]+suffix)
 rows=[]; step=round(sr/hop*1.5)
 for i in range(0,c.shape[1],step):
  v=c[:,i:i+step].mean(axis=1); scores=np.array(templates)@v/(np.linalg.norm(v)+1e-9); j=int(scores.argmax()); rows.append({'Inicio (s)':round(i*hop/sr,2),'Acorde':names[j] if scores[j]>.55 else '—','Similitud':round(float(scores[j]),2)})
 return rows
st.subheader(u.name)
st.audio(data)
@st.cache_data(show_spinner=False)
def waveform(path):
 y,sr=librosa.load(path,sr=8000,mono=True)
 if not len(y): return []
 return [float(np.max(np.abs(x))) if len(x) else 0 for x in np.array_split(y,420)]
def wave(path):
 vals=waveform(str(path)); peak=max(vals,default=1) or 1
 bars=''.join(f'<line x1="{i*2}" x2="{i*2}" y1="{28-25*v/peak}" y2="{28+25*v/peak}" stroke="#35d9bd" stroke-width="1.3"/>' for i,v in enumerate(vals))
 st.markdown(f'<svg viewBox="0 0 840 56" width="100%" style="background:#18232c;border-radius:8px">{bars}</svg>',unsafe_allow_html=True)
wave(p)
a,b,c=st.tabs(['Separar instrumentos','Acordes','Tablaturas'])
with a:
 out=folder/'htdemucs_6s'/p.stem
 mode=st.radio('Calidad de separación', ['Rápida','Equilibrada','Máxima'],index=1,horizontal=True)
 if st.button('Separar seis instrumentos'):
  with st.spinner('Procesando localmente. Puede tardar varios minutos…'):
   run=subprocess.run([sys.executable,'-m','demucs','-n','htdemucs_6s','-d','cpu','--segment','7','--shifts',str({'Rápida':0,'Equilibrada':1,'Máxima':2}[mode]),'-j','1','-o',str(folder),str(p)],capture_output=True,text=True)
  if run.returncode: st.error(run.stderr[-4000:])
  else: st.success('Separación terminada')
 stems=sorted(out.glob('*.wav'))
 labels={'guitar':'Guitarra','vocals':'Voz','bass':'Bajo','drums':'Batería','piano':'Piano','other':'Otros'}
 for s in stems:
  with st.container(border=True):
   left,right=st.columns([1,4]); left.markdown('**'+labels.get(s.stem,s.stem)+'**'); left.download_button('WAV',s.read_bytes(),s.name,key=str(s));
   with right:
    wave(s); st.audio(str(s))
 if stems:
  st.subheader('Mezclador y práctica'); tracks=[]; sr=None
  for s in stems:
   gain=st.slider(labels.get(s.stem,s.stem),0,100,100 if s.stem=='guitar' else 0,key='mix'+s.stem)/100
   y,sr=sf.read(s,always_2d=True,dtype="float32");
   if not tracks: tracks.append(np.zeros_like(y))
   tracks[0] += y*gain
  mix=tracks[0]; duration=len(mix)/sr
  start,end=st.slider('Fragmento (segundos)',0.0,float(duration),(0.0,float(duration)))
  speed=st.slider('Velocidad',0.5,1.25,1.0,0.05)
  if st.button('Preparar fragmento'):
   segment=mix[int(start*sr):int(end*sr)]
   if len(segment)<2048: st.warning('Selecciona un fragmento más largo.')
   else:
    if speed!=1: segment=librosa.effects.time_stretch(segment.T,rate=speed).T
    buf=io.BytesIO(); sf.write(buf,np.clip(segment,-1,1),sr,format='WAV'); st.session_state['mix'+str(p)]=buf.getvalue()
  if 'mix'+str(p) in st.session_state: st.audio(st.session_state['mix'+str(p)],loop=True)
with b:
 source=st.selectbox('Pista para analizar',[p]+stems,format_func=lambda x:labels.get(x.stem,x.name))
 if st.button('Detectar acordes'):
  with st.spinner('Analizando armonía…'): st.session_state['ch'+str(source)]=chords(str(source))
 rows=st.session_state.get('ch'+str(source))
 if rows:
  st.dataframe(rows,use_container_width=True); buf=io.StringIO(); w=csv.DictWriter(buf,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows); st.download_button('Exportar acordes CSV',buf.getvalue(),'acordes.csv')
 st.caption('Estimación por ventanas de 1,5 segundos; la similitud no es una probabilidad. Puede confundir inversiones y extensiones.')
with c:
 guitar=out/'guitar.wav'; target=st.selectbox('Pista para transcribir',([guitar,p] if guitar.exists() else [p]),format_func=lambda x:x.name)
 if st.button('Transcribir notas'):
  try:
   from basic_pitch.inference import predict
   with st.spinner('Detectando notas con Basic Pitch…'):
    _,midi,events=predict(str(target)); midi.write(str(folder/'notas.mid'))
   rows=[]; previous=3
   for start,end,pitch,amp,*_ in sorted(events):
    options=[(abs(pitch-open_note-previous),s,pitch-open_note) for s,open_note in enumerate([64,59,55,50,45,40],1) if 0<=pitch-open_note<=22]
    if options:
     _,s,f=min(options); previous=f; rows.append({'Inicio (s)':round(start,3),'Fin (s)':round(end,3),'MIDI':pitch,'Cuerda':s,'Traste':f})
   st.session_state['tab'+str(target)]=rows
  except Exception as e: st.error(str(e))
 rows=st.session_state.get('tab'+str(target),[])
 if rows:
  st.dataframe(rows,use_container_width=True)
  lines=[]
  for i in range(0,len(rows),16):
   group=rows[i:i+16]; lines.append(f"{group[0]['Inicio (s)']:.2f} s")
   for s,name in enumerate(['e','B','G','D','A','E'],1): lines.append(name+'|'+''.join((str(r['Traste']) if r['Cuerda']==s else '-').ljust(3,'-') for r in group)+'|')
   lines.append('')
  tab='\n'.join(lines); st.code(tab); st.download_button('Guardar tablatura',tab,'tablatura.txt'); st.download_button('Guardar MIDI',(folder/'notas.mid').read_bytes(),'notas.mid')
 st.caption('Digitación sugerida en afinación estándar. La tablatura ordena notas por inicio, no representa duración ni digitación polifónica exacta.')
