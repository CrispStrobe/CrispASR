require ARGV.fetch(0)
s = Whisper::CrispASR::Session
c = Whisper::CrispASR
chat = Whisper::CrispASR::Chat
$count=0
def eq(a,b)
  raise "#{a.inspect} != #{b.inspect}" unless a == b
  $count += 1
end
def reject(type)
  begin
    yield
  rescue type
    $count += 1
    return
  end
  raise "expected #{type}"
end
# Original 29 assertions (preserve exact semantics)
h=s.open('probe',1)
s.set_speaker_id(h,0);s.set_speaker_id(h,1)
[-1,2,2147483647].each do |id|
  begin
    s.set_speaker_id(h,id)
    raise "invalid speaker ID accepted: #{id}"
  rescue RuntimeError => error
    eq(error.message.include?('rc=-2'),true)
  end
end
a=s.transcribe(h,[1,3,5]).first
eq(a[:speaker],'speaker');eq(a[:words].first[:alts].first[:text],'alternative')
eq(a[:tokens].size,2);eq(a[:tokens].map{|x|x[:raw_text]}.join.force_encoding('UTF-8'),'日')
eq(a[:tokens].first[:raw_text].bytes,[0xe6,0x97]);eq(a[:tokens].first[:text].valid_encoding?,true)
eq(a[:tokens].first[:t0],-1);eq(a[:tokens].first[:has_timestamps],false)
eq(a[:t0],123)
eq(s.transcribe_chunked(h,[1,3,5],0,-1,nil).first[:tokens].size,2)
eq(s.transcribe_with_logits(h,[1,3,5]).first.first[:tokens].size,2)
b=s.transcribe_vad_lang(h,[1,3,5],'vad',nil).first
eq(b[:tokens].size,2);eq(b[:tokens].first[:t0],-1.0);eq(b[:t0],1.23)
piano=s.piano(h,[1,3,5]);eq(piano[:programs],[24,128]);eq(piano[:notes].size,8)
eq(s.chords(h,[1,3,5],16000)[:names],['C','Am'])
eq(s.beats(h,[1,3,5])[:tempo_bpm],123.5)
eq(s.tab(h,[1,3,5],22050)[:emissions].size,12)
eq(s.convert(h,[0,1,2,3,4,5],[1,3,5],0),[0,1,2,3,4,5])
reject(ArgumentError){s.convert(h,[0,1,2,3,4,5],[1,3,5],0,[0]*5,nil)}
s.set_voice_samples(h,[1,3,5],24000,'hello');s.set_tts_cfg_scale(h,1.5)
reject(RuntimeError){s.set_tts_cfg_scale(h,2)}
s.tada_set_makeref_models(h,nil,nil)
eq(s.get_disclaimer_pcm(h).first,0.25)
chunks=[];s.synthesize_streaming(h,'a. b.'){|pcm,last|chunks << [pcm,last]}
eq(chunks.size,2);eq(chunks.first.first.first,0.25);eq(chunks.last.first.first,0.75);eq(chunks.last.last,true)
reject(RuntimeError){s.synthesize_streaming(h,'test'){|pcm,last|s.close(h)}}
reject(ArgumentError){s.synthesize_streaming(h,'test'){|pcm,last|raise ArgumentError,'sentinel'}}
# New: audio load family
al=c.audio_load('probe.wav')
eq(al[:pcm].size,3);eq(al[:sample_rate],16000);eq(al[:n_samples],3)
al2=c.audio_load_at_rate('probe.wav', 24000)
eq(al2[:sample_rate],24000)
st=c.audio_load_stereo('probe.wav')
eq(st[:channels],2);eq(st[:left].size,3);eq(st[:right].size,3)
eq(st[:sample_rate],16000)
eq(c.audio_resample([0.1,0.2,0.3], 16000, 24000).size,3)
eq(c.audio_resample([], 16000, 24000).size,0)
eq(c.pcm_to_wav([0.1,0.2,0.3], 16000).bytesize,4)
eq(c.pcm_to_wav([0.1,0.2,0.3], 16000).encoding, Encoding::ASCII_8BIT)
cs=c.c2pa_sign("W\0AV", "audio/wav")
eq(cs.bytesize,4)
c.test_audio_mode(3)
eq(c.audio_load_stereo('probe')[:channels],1)
eq(c.test_audio_live,0)
c.test_audio_mode(2)
reject(RuntimeError){c.audio_load('probe')}
reject(RuntimeError){c.audio_load_at_rate('probe',24000)}
reject(RuntimeError){c.audio_load_stereo('probe')}
reject(RuntimeError){c.audio_resample([1,3,5],16000,24000)}
eq(c.test_audio_live,0)
c.test_audio_mode(0)
eq(s.kokoro_lang_is_german('de'),true);eq(s.kokoro_lang_is_german('en'),false)
eq(s.kokoro_lang_has_native_voice('en'),true);eq(s.kokoro_lang_has_native_voice('xx'),false)
c.watermark_load_model('probe.gguf')
eq(c.c_api_version, 'probe')
eq(c.dart_helpers_version, 'probe-helpers')
c.set_gpu_backend('cuda')
eq(c.backend_caps('foo'), 'asr,tts')
eq(c.backend_caps_list.include?('foo'), true)
eq(c.detect_backends_from_gguf('probe.gguf').include?('foo'), true)
eq(s.backend(h), 'probe')
sc=s.score_texts(h,[1,3,5],'en',nil,['hello','hi'])
eq(sc[:logprobs][0], -0.0);eq(sc[:n_tokens][1], 2)
s.set_pcm_sample_rate(h, 24000)
s.set_parakeet_att_context(h, 10, 20)
s.set_grammar_strict(h, true)
s.set_grammar_strict(h, false)
# truecase/pcs/punc
punc=s.punc_init('model')
eq(s.punc_process(punc,'hello'), 'Hello!')
s.punc_free(punc)
tc=s.truecase_init('model')
eq(s.truecase_process(tc,'hello'), "Gr\u00FC\u00DFe")
s.truecase_free(tc)
pcs=s.pcs_init('model')
eq(s.pcs_process(pcs,'hello'), 'Hello.')
s.pcs_free(pcs)
c.free_aligner_cache
# streamed polling
eq(c.get_progress, 42)
c.reset_progress
eq(c.get_progress, -1)
eq(c.get_streamed_segment_count, 2)
segs=c.drain_streamed_segments
eq(segs.first[:text], 'Japanese')
eq(segs.first[:t0], 123)
c.reset_streamed_segments
eq(c.get_streamed_segment_count, 0)
eq(c.get_streamed_token_count, 3)
toks=c.drain_streamed_tokens
eq(toks.size, 4)
eq(toks.first.bytes, [0xe6,0x97])
c.reset_streamed_tokens
eq(c.get_streamed_token_count, 0)
# callbacks with rooted lifetime and exception deferral
progress=[]
s.set_progress_callback(h, ->(pr,t){ progress << [pr,t] })
seg_texts=[]
s.set_segment_callback(h, ->(txt,t0,t1,idx){ seg_texts << txt })
tok_texts=[]
s.set_token_callback(h, ->(txt,idx){ tok_texts << txt })
s.transcribe(h,[1,3,5])
eq(progress.first, [3,6])
eq(seg_texts.first, "日本語")
eq(tok_texts.first.bytes, [0xe6,0x97])
s.set_segment_callback(h, ->(txt,t0,t1,idx){ raise ArgumentError,'cb err' })
reject(ArgumentError){ s.transcribe(h,[1,3,5]) }
s.set_segment_callback(h, ->(txt,t0,t1,idx){ s.transcribe(h,[1,3,5]) })
reject(RuntimeError){ s.transcribe(h,[1,3,5]) }
s.set_progress_callback(h, ->(_,_){raise ArgumentError,'root sentinel'})
s.set_segment_callback(h, ->(*_){GC.compact if GC.respond_to?(:compact)})
begin
  s.transcribe(h,[1,3,5]);raise 'missing deferred callback error'
rescue ArgumentError => e
  eq(e.message,'root sentinel')
end
s.set_progress_callback(h, nil)
s.set_segment_callback(h, nil)
s.set_token_callback(h, nil)
s.transcribe(h,[1,3,5])
# cross-session / unknown-userdata routing — verifies exact userdata dispatch, no fallback scan

# Real wrapper opens receive distinct native fixture handles.
h1=s.open('probe',1); h2=s.open('probe',1)
prog1=[]; prog2=[]
s.set_progress_callback(h1, ->(a,b){ prog1 << [a,b] })
s.set_progress_callback(h2, ->(a,b){ prog2 << [a,b] })
c.test_invoke_progress(h1)
eq(prog1.size,1); eq(prog2.size,0)
eq(prog1.first, [3,6])
c.test_invoke_progress(h2)
eq(prog2.size,1); eq(prog1.size,1)
# unknown/null must not route to other session (no fallback scan)
c.test_invoke_progress(nil)
eq(prog1.size,1); eq(prog2.size,1)
c.test_invoke_progress(0xdeadbeef)
eq(prog1.size,1); eq(prog2.size,1)
seg1=[]; seg2=[]
s.set_segment_callback(h1, ->(txt,t0,t1,idx){ seg1 << txt })
s.set_segment_callback(h2, ->(txt,t0,t1,idx){ seg2 << txt })
c.test_invoke_segment(h1)
eq(seg1.first, "日本語"); eq(seg2.size,0)
c.test_invoke_segment(h2)
eq(seg2.first, "日本語"); eq(seg1.size,1)
c.test_invoke_segment(nil)
eq(seg1.size,1); eq(seg2.size,1)
c.test_invoke_segment(0xdeadbeef)
eq(seg1.size,1); eq(seg2.size,1)
tok1=[]; tok2=[]
s.set_token_callback(h1, ->(txt,idx){ tok1 << txt })
s.set_token_callback(h2, ->(txt,idx){ tok2 << txt })
c.test_invoke_token(h1)
eq(tok1.first.bytes, [0xe6,0x97]); eq(tok2.size,0)
c.test_invoke_token(h2)
eq(tok2.first.bytes, [0xe6,0x97]); eq(tok1.size,1)
c.test_invoke_token(nil)
eq(tok1.size,1); eq(tok2.size,1)
c.test_invoke_token(0xdeadbeef)
eq(tok1.size,1); eq(tok2.size,1)
# rooted callbacks clear on close — delete hash entry, no dangling proc
s.close(h1)
c.test_invoke_progress(h1)
eq(prog1.size,1)
c.test_invoke_segment(h1)
eq(seg1.size,1)
c.test_invoke_token(h1)
eq(tok1.size,1)
s.close(h2)
c.test_invoke_progress(h2)
eq(prog2.size,1)
# pitch / separate / dimension getters
pit=s.pitch(h,[1,3,5],10.0)
eq(pit[:n_frames], 3)
eq(pit[:frames].size, 9)
eq(pit[:sample_rate], 16000)
eq(s.pitch_n_frames(h), 3)
eq(s.pitch_sample_rate(h), 16000)
eq(s.pitch_frame(h,0)[:f0_hz], 440.0)
sep=s.separate(h,[0,0,0,0])
eq(sep[:n_stems], 2)
eq(sep[:stems].first[:name], 'vocals')
eq(sep[:sample_rate], 44100)
eq(s.separate_n_stems(h), 2)
eq(s.separate_sample_rate(h), 44100)
eq(s.convert_sample_rate(h), 40000)
eq(s.piano_n_notes(h), 2)
eq(s.chords_n_spans(h), 2)
eq(s.beats_n_events(h), 2)
eq(s.tab_n_frames(h), 2)
# transcribe_vad plain (without lang) with full metadata
vad_plain=s.transcribe_vad(h,[1,3,5],'vad')
eq(vad_plain.first[:tokens].size,2)
eq(vad_plain.first[:has_timestamps], true)
# registry / cache
rl=c.registry_lookup('backend')
eq(rl[:filename], 'model.gguf')
rbf=c.registry_lookup_by_filename('model.gguf')
eq(rbf[:url], 'https://example/model.gguf')
rbi=c.registry_default_bundle_info('backend')
eq(rbi[:license], 'MIT')
rba=c.registry_default_bundle_artifact('backend',0)
eq(rba[:kind], 0)
eq(c.cache_ensure_file('f','https://example'), '/tmp/cache/model.gguf')
eq(c.cache_dir(nil), '/tmp/cache')
opts={method:4,n_threads:7,slice_t0_cs:125,model_path:'model',embedder_path:'embed'}
diarized=c.diarize_turns([1,3,5],nil,[{t0:100,t1:200}],opts)
eq(diarized[:segments].first[:speaker],1);eq(diarized[:turns].first[:t0],125);eq(diarized[:turns].first[:t1],250);eq(diarized[:turns].first[:speaker],2)
reject(ArgumentError){c.diarize_turns([1,3,5],[1],[{t0:100,t1:200}],opts)}
# chat full lifecycle with structured copying and no fake streaming
ch=chat.open('probe', {'n_ctx'=>2048,'n_gpu_layers'=>0,'use_mmap'=>false,'chat_template'=>'gemma'})
eq(chat.template_name(ch), 'probe-template')
eq(chat.n_ctx(ch), 2048)
eq(chat.count_tokens(ch, [{'role'=>'user','content'=>'hi'}]), 2)
eq(chat.memory_estimate('probe',{}) > 0, true)
eq(chat.ai_disclosure_text.include?('AI'), true)
msg=[{'role'=>'system','content'=>'system'},{'role'=>'user','content'=>"日本語"}]
eq(chat.generate(ch, msg, {'max_tokens'=>7,'seed'=>0xffffffff, 'temperature'=>0.8,'stop'=>['STOP']}), "Gr\u00FC\u00DFe")
eq(chat.generate(ch,msg,{prefill_only:true}), "")
chunks2=[]
chat.generate_stream(ch, msg, {}){|chunk| GC.compact if GC.respond_to?(:compact); chunks2 << chunk }
eq(chunks2.size, 2)
eq(chunks2.first.bytes, [0xe6,0x97])
eq(chunks2.last, "\xA5")
# chat token/abort exception + errorcode 40 survival
begin
  chat.generate_stream(ch, msg, {}){|chunk| raise ArgumentError,'tok boom' }
  raise "should fail"
rescue ArgumentError => e
  eq(e.message, 'tok boom')
end
chat.set_abort_callback(ch, ->{ raise ArgumentError,'abort boom' })
begin
  chat.generate(ch, msg, {})
  raise "should fail"
rescue ArgumentError => e
  eq(e.message, 'abort boom')
end
first_abort=0;second_abort=0
other_chat=chat.open('probe',{})
chat.set_abort_callback(ch, ->{first_abort+=1;GC.compact if GC.respond_to?(:compact);true})
chat.set_abort_callback(other_chat, ->{second_abort+=1;true})
GC.compact if GC.respond_to?(:compact)
eq(chat.generate(ch,msg,{}),"Grüße");eq(first_abort,1);eq(second_abort,0)
eq(chat.generate(other_chat,msg,{}),"Grüße");eq(first_abort,1);eq(second_abort,1)
chat.close(other_chat)
chat.set_abort_callback(ch, nil)
begin
  chat.generate_stream(ch, msg, {}){|chunk| raise ArgumentError,'tok2' }
  raise "should fail"
rescue ArgumentError => e
  eq(e.message, 'tok2')
end
aborted=false
chat.set_abort_callback(ch, ->{ aborted=true; false })
begin
  chat.generate(ch, msg, {})
rescue RuntimeError => e
  eq(e.message.include?('cancelled'), true)
end
eq(aborted, true)
chat.set_abort_callback(ch, nil)
chat.reset(ch)
chat.close(ch)
begin
  chat.generate(ch, msg, {})
  raise "should fail"
rescue RuntimeError, StandardError
  $count += 1
end
reject(RuntimeError){ chat.open('fail', {}) }
pk=s.parakeet_init('probe',4,true)
pkr=s.parakeet_transcribe(pk,[1,3,5],123)
eq(pkr[:text],'日本語');eq(pkr[:words].first[1],223);eq(pkr[:tokens].first[1],-1);eq(pkr[:tokens].map{|t|t[0]}.join.force_encoding('UTF-8'),'日')
reject(ArgumentError){s.parakeet_transcribe(pk,[1,3,5],'de')}
s.parakeet_free(pk)
titan=s.titanet_init('probe',4)
eq(s.titanet_embed(titan,[1,3,5]).length,192);eq(s.titanet_embed(titan,[1,3,5]).first,1.0)
[2,3,4].each{|mode|c.test_titanet_mode(mode);reject(RuntimeError){s.titanet_embed(titan,[1,3,5])}};c.test_titanet_mode(0)
reject(ArgumentError){s.titanet_embed(titan,[])};s.titanet_free(titan);s.titanet_free(titan)
eq(c.test_titanet_live,0);reject(RuntimeError){s.titanet_embed(titan,[1,3,5])}
s.parakeet_free(pk);reject(RuntimeError){s.parakeet_transcribe(pk,[1,3,5],0)}
eq(s.titanet_cosine_sim([1,0],[1,0]),1.0)
eq(s.titanet_cosine_sim([],[]),0.0)
reject(ArgumentError){s.titanet_cosine_sim([1],[1,0])}
coercible=Object.new
def coercible.to_f; GC.compact if GC.respond_to?(:compact);1.0;end
eq(s.titanet_cosine_sim([coercible,0],[1,0]),1.0)
c.test_db_mode(0)
reject(RuntimeError){s.speaker_db_open('db','Alice,Bob',false)}
reject(ArgumentError){s.speaker_db_open('db','',true)}
db=s.speaker_db_open('db','Alice,Bob',true);eq(s.speaker_db_count(db),2);eq(c.test_db_live,1)
matched=s.speaker_db_match(db,[1,3,5],0.5);eq(matched,['アリス',0.875])
c.test_db_mode(1);eq(s.speaker_db_match(db,[1,3,5],-1),[nil,-1.0]);eq(matched.first,'アリス');c.test_db_mode(0)
reject(ArgumentError){s.speaker_db_match(db,[],0.5)};reject(ArgumentError){s.speaker_db_match(db,[1,3,5],Float::NAN)}
reject(RuntimeError){s.speaker_db_enroll('db','Alice',[1,3,5],false)};reject(ArgumentError){s.speaker_db_enroll('db','Alice',[],true)}
eq(s.speaker_db_enroll('db','Alice',[1,3,5],true),true)
c.test_db_mode(3);reject(RuntimeError){s.speaker_db_enroll('db','Alice',[1,3,5],true)}
c.test_db_mode(2);reject(RuntimeError){s.speaker_db_open('db','Alice,Bob',true)};c.test_db_mode(0)
s.speaker_db_free(db);s.speaker_db_free(db);eq(c.test_db_live,0)
reject(RuntimeError){s.speaker_db_count(db)};reject(RuntimeError){s.speaker_db_match(db,[1,3,5],0.5)}
c.test_speaker_mode(0)
# Ruby coercion can re-enter initialize before native ownership is installed.
recursive=c::SpeakerEmbedder.allocate
model=Object.new
model.define_singleton_method(:to_str){recursive.send(:initialize,'embed',2,'cache');'embed'}
reject(RuntimeError){recursive.send(:initialize,model,2,'cache')};eq(c.test_speaker_live,1);recursive.close;eq(c.test_speaker_live,0)
recursive_cache=c::PyannoteCache.allocate
cache_model=Object.new
cache_model.define_singleton_method(:to_str){recursive_cache.send(:initialize,[1,3,5],'seg',2);'seg'}
reject(RuntimeError){recursive_cache.send(:initialize,[1,3,5],cache_model,2)};eq(c.test_speaker_live,1);recursive_cache.close;eq(c.test_speaker_live,0)
embedder=c::SpeakerEmbedder.new('embed',2,'cache');eq(embedder.dimension,3);eq(embedder.name,'generic')
embedding=embedder.embed([1,3,5]);eq(embedding,[0.25,0.5,0.75]);eq(c.test_speaker_live,1)
reject(ArgumentError){embedder.embed([])};reject(ArgumentError){c::SpeakerEmbedder.new('embed',0)}
[2,4].each{|mode|c.test_speaker_mode(mode);reject(RuntimeError){embedder.embed([1,3,5])}};c.test_speaker_mode(0)
embedder.close;embedder.close;eq(c.test_speaker_live,0);eq(embedding,[0.25,0.5,0.75])
reject(RuntimeError){embedder.dimension};reject(RuntimeError){embedder.name};reject(RuntimeError){embedder.embed([1,3,5])}
[1,3].each{|mode|c.test_speaker_mode(mode);reject(RuntimeError){c::SpeakerEmbedder.new('embed',2,'cache')}};eq(c.test_speaker_live,0);c.test_speaker_mode(0)
eq(c.speaker_cluster([[coercible,2,3],[4,5,6]],0.6,3)[:labels],[0,1])
cluster=c.speaker_cluster([[1,2,3],[4,5,6]],0.6,3);eq(cluster[:labels],[0,1]);eq(cluster[:n_clusters],2)
reject(ArgumentError){c.speaker_cluster([],0.6,3)};reject(ArgumentError){c.speaker_cluster([[1],[2,3]],0.6,3)}
reject(ArgumentError){c.speaker_cluster([[1,2,3],[4,5,6]],Float::NAN,3)}
[2,6].each{|mode|c.test_speaker_mode(mode);reject(RuntimeError){c.speaker_cluster([[1,2,3],[4,5,6]],0.6,3)}};c.test_speaker_mode(0)
cache=c::PyannoteCache.new([1,3,5],'seg',2);grid=[{t0:125,t1:175},{'t0'=>200,'t1'=>250}]
time=Object.new;def time.to_int;GC.compact if GC.respond_to?(:compact);125;end
eq(cache.apply([{t0:time,t1:175},{t0:200,t1:250}],125).first[:speaker],2)
assigned=cache.apply(grid,125);eq(assigned.map{|s|s[:speaker]},[2,1]);eq(assigned.first[:t0],125);eq(grid.first[:speaker],nil)
[2,5].each{|mode|c.test_speaker_mode(mode);reject(RuntimeError){cache.apply(grid,125)}};c.test_speaker_mode(0)
reject(ArgumentError){cache.apply([],125)};reject(ArgumentError){cache.apply([{t0:200,t1:100}],125)}
cache.close;cache.close;eq(c.test_speaker_live,0);eq(assigned.last[:speaker],1);reject(RuntimeError){cache.apply(grid,125)}
c.test_speaker_mode(1);reject(RuntimeError){c::PyannoteCache.new([1,3,5],'seg',2)};c.test_speaker_mode(0)
reject(ArgumentError){c::PyannoteCache.new([],'seg',2)}
def orphan_speaker(c); c::SpeakerEmbedder.new('embed');nil;end
# A live VM/C stack may conservatively retain a discarded temporary. Allocate
# on a finished thread so collection does not depend on register/stack contents.
Thread.new { orphan_speaker(c) }.join;GC.start;eq(c.test_speaker_live,0)
# A reused native address must never resurrect an already closed public handle.
c.test_titanet_reuse(true)
stale=s.titanet_init('probe',4);s.titanet_free(stale)
replacement=s.titanet_init('probe',4)
reject(RuntimeError){s.titanet_embed(stale,[1,3,5])}
s.titanet_free(stale);eq(c.test_titanet_live,1);eq(s.titanet_embed(replacement,[1,3,5]).length,192)
s.titanet_free(replacement);eq(c.test_titanet_live,0);c.test_titanet_reuse(false)
borrowed_titan=s.titanet_init('probe',4)
value=Object.new;value.define_singleton_method(:to_f){s.titanet_free(borrowed_titan);1.0}
reject(RuntimeError){s.titanet_embed(borrowed_titan,[value,3,5])};eq(c.test_titanet_live,1)
s.titanet_free(borrowed_titan);eq(c.test_titanet_live,0)
eq(s.method(:transcribe).arity,2);eq(s::Stream.method(:open).arity,6);eq(chat.method(:generate).arity,3)
s.singleton_class.alias_method(:contract_transcribe,:transcribe)
eq(s.contract_transcribe(h,[1,3,5]).first[:tokens].length,2)
s.singleton_class.remove_method(:contract_transcribe)
loadopts={n_threads:7,use_gpu:false,verbosity:2,flash_attn:false,n_gpu_layers:3}
loaded=s.open_with_options('probe','probe',loadopts)
opts_tokens=[];s.set_token_callback(loaded,->(text,index){opts_tokens << text})
eq(s.transcribe(loaded,[1,3,5]).first[:tokens].length,2);eq(opts_tokens.first.bytes,[0xe6,0x97])
s.close(loaded);s.close(loaded);reject(RuntimeError){s.transcribe(loaded,[1,3,5])}
reloaded=s.open_with_options('probe','probe',loadopts)
reject(RuntimeError){s.transcribe(loaded,[1,3,5])};s.close(loaded)
eq(s.transcribe(reloaded,[1,3,5]).first[:tokens].length,2);s.close(reloaded)
reject(ArgumentError){s.open_with_options('probe','probe',loadopts.merge(n_gpu_layers:-2))}
reject(ArgumentError){s.open_with_options('probe','probe',loadopts.merge(use_gpu:0))}
reject(RuntimeError){s.open_with_options('probe','probe',loadopts.merge(n_threads:6))}
path=Object.new;def path.to_str;GC.compact if GC.respond_to?(:compact);'probe';end
compacted_chat=chat.open(path,{chat_template:'gemma',n_ctx:2048,n_gpu_layers:0,use_mmap:false})
eq(chat.n_ctx(compacted_chat),2048);chat.close(compacted_chat)
c.test_text_mode(0)
[[:punc,'Hello!'],[:truecase,'Grüße'],[:pcs,'Hello.']].each do |prefix,expected|
  handle=s.send("#{prefix}_init",'model');eq(s.send("#{prefix}_process",handle,'hello'),expected);eq(c.test_text_live,0)
  c.test_text_mode(2);eq(s.send("#{prefix}_process",handle,''),'');eq(c.test_text_live,0)
  c.test_text_mode(1);reject(RuntimeError){s.send("#{prefix}_process",handle,'hello')};c.test_text_mode(0)
  reject(RuntimeError){s.send(prefix==:punc ? 'truecase_free':'punc_free',handle)}
  value=Object.new;value.define_singleton_method(:to_str){s.send("#{prefix}_free",handle);'hello'}
  reject(RuntimeError){s.send("#{prefix}_process",handle,value)}
  s.send("#{prefix}_free",handle);s.send("#{prefix}_free",handle);eq(c.test_text_contexts,0)
  reject(RuntimeError){s.send("#{prefix}_process",handle,'hello')}
end
coercion_owner=s.open('probe',1)
number=Object.new
number.define_singleton_method(:to_f){s.close(coercion_owner);0.5}
before_temperature=c.test_temperature_calls
reject(RuntimeError){s.set_temperature(coercion_owner,number,0)}
eq(c.test_temperature_calls,before_temperature)
s.set_temperature(coercion_owner,0.5,0);eq(c.test_temperature_calls,before_temperature+1)
eq(s.backend(coercion_owner),'probe')
reject(RuntimeError) do
  s.synthesize_streaming(coercion_owner,'test') do |pcm,last|
    Thread.new{Thread.current.report_on_exception=false;s.close(coercion_owner)}.value
  end
end
eq(s.backend(coercion_owner),'probe');s.close(coercion_owner)
streams=Whisper::CrispASR::Session::Stream
owner=s.open('probe',1)
stream=streams.open(owner,100,300,10,nil,false)
eq(streams.feed(stream,[1,3,5]),1);eq(streams.get_text(stream)[:text],'ok');streams.set_live_decode(stream,true)
eq(s.stream_kind(owner),1)
coercion=Object.new;coercion.define_singleton_method(:to_f){s.close(owner);1.0}
reject(RuntimeError){streams.feed(stream,[coercion,3,5])}
closed_before=c.test_stream_closes
s.close(owner)
eq(c.test_parent_close_violations,0);eq(c.test_stream_closes,closed_before+1)
s.close(owner);streams.close(stream);eq(c.test_stream_closes,closed_before+1)
reject(RuntimeError){streams.feed(stream,[1,3,5])};reject(RuntimeError){streams.get_text(stream)}
reject(RuntimeError){streams.flush(stream)};reject(RuntimeError){streams.set_live_decode(stream,true)}
reject(RuntimeError){s.stream_kind(owner)}
reject(RuntimeError){s.transcribe(owner,[1,3,5])};reject(RuntimeError){streams.open(owner,100,300,10,nil,false)}
s.close(h)
require 'timeout'
mic=c::Mic
received=Queue.new
mh=mic.open(16000,1) do |pcm|
  error=nil
  begin
    mic.close(mh)
  rescue => e
    error=e.class.to_s
  end
  received << [pcm,error]
end
mic.start(mh)
value=Timeout.timeout(5){received.pop}
eq(value,[[0.0,0.5,-0.25],nil])
eq(c.test_mic_live,0)
mic.close(mh);mic.stop(mh);reject(RuntimeError){mic.start(mh)}
eq(mic.default_device_name,'Probe microphone')
reject(RuntimeError){mic.start(999999999)};eq(mic.close(999999999),nil)
reject(TypeError){mic.start('invalid')}
reject(ArgumentError){mic.open(16000,1)}
reject(ArgumentError){mic.open(0,1){}}
reject(ArgumentError){mic.open(16000,3){}}
bad_rate=Object.new;def bad_rate.to_int; raise ArgumentError,'coercion';end
reject(ArgumentError){mic.open(bad_rate,1){}};eq(c.test_mic_live,0)
c.test_mic_mode(1);reject(RuntimeError){mic.open(16000,1){}};eq(c.test_mic_live,0);c.test_mic_mode(0)
# Native start/stop failures leave a closeable owner.
mh=mic.open(16000,1){}
c.test_mic_mode(2);reject(RuntimeError){mic.start(mh)};eq(c.test_mic_live,1)
c.test_mic_mode(3);reject(RuntimeError){mic.stop(mh)};c.test_mic_mode(0)
closed_before=c.test_mic_closes
[Thread.new{mic.close(mh)},Thread.new{mic.close(mh)}].each(&:join)
eq(c.test_mic_closes,closed_before+1);eq(c.test_mic_live,0)
# Ring overflow drops the oldest chunks. Callback exceptions/GC do not kill delivery.
c.test_mic_mode(4);finished=Queue.new;seen=[]
mh=mic.open(16000,1) do |pcm|
  seen << pcm.first
  if seen.length==1
    GC.start;GC.compact if GC.respond_to?(:compact)
    raise 'expected callback error'
  end
  finished << true if seen.length==31
end
mic.start(mh);Timeout.timeout(5){finished.pop};mic.stop(mh);mic.close(mh)
eq(seen,(39..69).map(&:to_f));eq(c.test_mic_live,0);eq(mic.test_live_chunks,0);c.test_mic_mode(0)
# Killing an idle pump still closes the native microphone and retires its ticket.
threads=Queue.new
mh=mic.open(16000,1){|pcm|threads << Thread.current};mic.start(mh)
pump=Timeout.timeout(5){threads.pop};pump.kill;pump.join
eq(c.test_mic_live,0);reject(RuntimeError){mic.start(mh)};mic.close(mh)
# New owners cannot revive old tickets, even if native malloc reuses an address.
old=mh;mh=mic.open(16000,1){};eq(mh==old,false);reject(RuntimeError){mic.start(old)}
mic.close(old);eq(c.test_mic_live,1);mic.close(mh);eq(c.test_mic_live,0)
# A copy/allocation exception after native dequeue must free the pending chunk.
c.test_mic_mode(5);calls=0
mh=mic.open(16000,1){calls+=1};pump=mic.test_pump(mh);pump.report_on_exception=false
mic.start(mh);reject(NoMemoryError){pump.join}
eq(calls,0);eq(c.test_mic_live,0);eq(mic.test_live_chunks,0)
reject(RuntimeError){mic.start(mh)};mic.close(mh);c.test_mic_mode(0)
puts "PASS Ruby ABI: #{$count} assertions"
