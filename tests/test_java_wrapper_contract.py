"""Real JNA calls through the Java wrapper into public-header C probes."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
JAR=os.environ.get('CRISPASR_JNA_JAR')
JAVA=os.environ.get('JAVA') or shutil.which('java')
PROGRAM=r'''
import io.github.ggerganov.whispercpp.*;
import com.sun.jna.*;
import java.util.*;
import java.nio.charset.StandardCharsets;
public class WrapperAbiProbe {
    public interface Probe extends Library {
        Probe INSTANCE=Native.load("crispasr",Probe.class);
        void probe_vad_expect(int n);int probe_vad_calls();void probe_mode(int n);void probe_mutate();int probe_live();int probe_invalid_frees();int probe_result_frees();int probe_calls();void probe_utility_mode(int mode);int probe_utility_live();
    }
    static int assertions;
    static void eq(Object a,Object b){if(!Objects.equals(a,b))throw new AssertionError(a+" != "+b);assertions++;}
    static void reject(Class<? extends Throwable> cls,Runnable fn) {
        try {fn.run();}catch(Throwable t){if(cls.isInstance(t)){assertions++;return;}throw t;}
        throw new AssertionError("expected "+cls.getName());
    }
    public static void main(String[] args) {
        float[] pcm={1,3,5};boolean old=args.length>0;
        try(CrispasrSession s=CrispasrSession.open("probe",4)) {
            s.setSpeakerId(0);s.setSpeakerId(1);
            for(int id:new int[]{-1,2,Integer.MAX_VALUE}) {
                try {s.setSpeakerId(id);throw new AssertionError("invalid speaker ID accepted: "+id);}
                catch(IllegalStateException error) {eq(error.getMessage().contains("rc=-2"),true);}
            }
            CrispasrSession.Segment a=s.transcribe(pcm)[0];
            eq(a.t0,123L);eq(a.words[0].t0,-1L);eq(a.words[0].hasTimestamps(),false);
            eq(a.words[0].alts[0].text,"alternative");eq(a.speaker,"speaker");
            if(old){eq(a.tokens.length,0);System.out.println("PASS Java older ABI: "+assertions);return;}
            eq(a.tokens.length,2);eq(a.tokens[0].rawText[0],(byte)0xe6);
            eq(a.tokens[0].hasTimestamps(),false);eq(a.tokens[1].t0,123L);
            eq(new String(new byte[]{a.tokens[0].rawText[0],a.tokens[0].rawText[1],a.tokens[1].rawText[0]},StandardCharsets.UTF_8),"日");
            eq(s.transcribeChunked(pcm,0,-1,null)[0].tokens.length,2);
            eq(s.transcribeVad(pcm,16000,"vad")[0].tokens.length,2);
            CrispasrSession.VadOptions vad=new CrispasrSession.VadOptions();
            vad.threshold=.375f;vad.minSpeechDurationMs=123;vad.minSilenceDurationMs=456;vad.speechPadMs=78;vad.chunkSeconds=9;vad.nThreads=2;
            Probe.INSTANCE.probe_vad_expect(1);eq(s.transcribeVad(pcm,16000,"vad","ja",vad)[0].tokens.length,2);
            Probe.INSTANCE.probe_vad_expect(2);eq(s.transcribeVad(pcm,16000,"vad","ja",null)[0].tokens.length,2);
            int vadCalls=Probe.INSTANCE.probe_vad_calls();
            reject(IllegalArgumentException.class,()->s.transcribeVad(new float[0],16000,"vad","ja",vad));
            reject(NullPointerException.class,()->s.transcribeVad(null,16000,"vad","ja",vad));
            reject(IllegalArgumentException.class,()->s.transcribeVad(pcm,0,"vad","ja",vad));
            vad.threshold=Float.NaN;reject(IllegalArgumentException.class,()->s.transcribeVad(pcm,16000,"vad","ja",vad));
            vad.threshold=.375f;vad.nThreads=0;reject(IllegalArgumentException.class,()->s.transcribeVad(pcm,16000,"vad","ja",vad));
            eq(Probe.INSTANCE.probe_vad_calls(),vadCalls);vad.nThreads=2;
            Probe.INSTANCE.probe_vad_expect(3);reject(IllegalStateException.class,()->s.transcribeVad(pcm,16000,"vad","ja",vad));
            Probe.INSTANCE.probe_vad_expect(1);int vadFrees=Probe.INSTANCE.probe_result_frees();
            s.setTokenCallback((x,i)->{throw new IllegalArgumentException("vad callback");});
            reject(IllegalArgumentException.class,()->s.transcribeVad(pcm,16000,"vad","ja",vad));
            eq(Probe.INSTANCE.probe_result_frees(),vadFrees+1);s.setTokenCallback(null);Probe.INSTANCE.probe_vad_expect(0);
            eq(s.transcribeWithLogits(pcm,null).segments[0].tokens.length,2);
            CrispasrSessionTasks.PianoResult piano=s.piano(pcm);eq(piano.programs[1],128);eq(piano.sampleRate,16000);
            eq(s.chords(pcm,16000).names[1],"Am");eq(s.beats(pcm).tempoBpm,123.5f);
            eq(s.backend(),"probe");eq(CrispasrSession.loadAudio("x").samples[2],.75f);
            eq(CrispasrSession.loadAudio("x",24000).sampleRate,24000);
            eq(CrispasrSession.loadAudioStereo("x").samples.length,6);
            Probe.INSTANCE.probe_utility_mode(3);eq(CrispasrSession.loadAudioStereo("x").sourceChannels,1);
            eq(CrispasrSession.resample(new float[0],16000,24000).length,0);
            eq(CrispasrSession.resample(pcm,16000,24000)[0],.25f);
            eq(CrispasrSession.pcmToWav(pcm,16000)[1],(byte)0);
            eq(Arrays.toString(CrispasrSession.signC2pa(new byte[]{1,0,2},"audio/wav",null,null)),"[1, 0, 2]");
            Probe.INSTANCE.probe_utility_mode(2);
            reject(IllegalStateException.class,()->CrispasrSession.loadAudio("x"));
            reject(IllegalStateException.class,()->CrispasrSession.loadAudioStereo("x"));
            reject(IllegalStateException.class,()->CrispasrSession.signC2pa(new byte[]{1},"audio/wav",null,null));
            eq(Probe.INSTANCE.probe_utility_live(),0);Probe.INSTANCE.probe_utility_mode(0);
            String translated=s.translateText("hello","en","ja");eq(translated,"こんにちは");eq(Probe.INSTANCE.probe_live(),0);
            Probe.INSTANCE.probe_mode(1);eq(s.translateText("","en","ja",0),"");Probe.INSTANCE.probe_mode(2);
            reject(IllegalStateException.class,()->s.translateText("hello","en","ja"));Probe.INSTANCE.probe_mode(0);
            reject(NullPointerException.class,()->s.translateText(null,"en","ja"));eq(translated,"こんにちは");
            eq(CrispasrSession.enhanceAudioRnnoise(pcm)[0],.5f);eq(pcm[0],1f);
            eq(CrispasrSession.enhanceAudioRnnoise(new float[0]).length,0);reject(NullPointerException.class,()->CrispasrSession.enhanceAudioRnnoise(null));
            Probe.INSTANCE.probe_utility_mode(2);reject(IllegalStateException.class,()->CrispasrSession.enhanceAudioRnnoise(pcm));
            Probe.INSTANCE.probe_utility_mode(4);reject(IllegalStateException.class,()->CrispasrSession.enhanceAudioRnnoise(pcm));Probe.INSTANCE.probe_utility_mode(0);
            CrispasrUtilities.TextProcessor punc=new CrispasrUtilities.TextProcessor("probe",CrispasrUtilities.TextProcessorKind.PUNCTUATION);
            eq(punc.process("hello"),"Hello!");eq(Probe.INSTANCE.probe_live(),0);
            Probe.INSTANCE.probe_mode(1);eq(punc.process(""),"");Probe.INSTANCE.probe_mode(2);
            reject(IllegalStateException.class,()->punc.process("hello"));Probe.INSTANCE.probe_mode(0);
            reject(NullPointerException.class,()->punc.process(null));punc.close();punc.close();reject(IllegalStateException.class,()->punc.process("hello"));
            eq(s.scoreTexts(pcm,new String[]{"one","two"},null,null).tokenCounts[1],3);
            eq(CrispasrSession.kokoroLanguageIsGerman("de"),true);eq(CrispasrSession.kokoroLanguageIsGerman("en"),false);
            eq(CrispasrSession.kokoroLanguageHasNativeVoice("en"),true);eq(CrispasrSession.kokoroLanguageHasNativeVoice("xx"),false);
            eq(CrispasrSession.cApiVersion(),"probe");eq(CrispasrSession.helperVersion(),"probe-helpers");
            eq(CrispasrSession.backendCapabilities("foo")[0],"asr");
            try(CrispasrUtilities.TextProcessor processor=new CrispasrUtilities.TextProcessor("probe",false)){eq(processor.process("test"),"Grüße");}
            try(CrispasrUtilities.TextProcessor processor=new CrispasrUtilities.TextProcessor("probe",true)){eq(processor.process("test"),"Hello.");}
            eq(s.tab(pcm,22050).emissions.length,12);eq(s.tab(pcm,22050).openMidi[1],45);
            Probe.INSTANCE.probe_mutate();eq(piano.notes[0],0f);
            eq(s.convert(new float[]{0,1,2,3,4,5},pcm,0).length,6);
            reject(IllegalArgumentException.class,()->s.convert(new float[6],pcm,0,new float[5],null));
            Probe.INSTANCE.probe_mode(1);
            eq(s.piano(pcm).notes.length,0);eq(s.chords(pcm,16000).spans.length,0);eq(s.beats(pcm).events.length,0);
            Probe.INSTANCE.probe_mode(0);
            s.setVoiceSamples(pcm,24000,"hello");s.setTtsCfgScale(1.5f);s.tadaSetMakerefModels(null,null);
            reject(IllegalStateException.class,()->s.setTtsCfgScale(2));
            eq(s.getDisclaimerPcm()[0],.25f);eq(Probe.INSTANCE.probe_live(),0);
            List<float[]> chunks=new ArrayList<>();s.synthesizeStreaming("two. chunks.",(x,last)->chunks.add(x));
            eq(chunks.size(),2);eq(chunks.get(0)[0],.25f);eq(chunks.get(1)[0],.75f);
            reject(IllegalStateException.class,()->s.synthesizeStreaming("test",(x,last)->s.close()));
            int before=Probe.INSTANCE.probe_result_frees();
            s.setTokenCallback((x,i)->{throw new IllegalArgumentException("sentinel");});
            reject(IllegalArgumentException.class,()->s.transcribe(pcm));
            eq(Probe.INSTANCE.probe_result_frees(),before+1);s.setTokenCallback(null);
            eq(Probe.INSTANCE.probe_live(),0);eq(Probe.INSTANCE.probe_invalid_frees(),0);
        }
        CrispasrSession owner=CrispasrSession.open("probe",1);
        CrispasrSession.Stream stream=owner.streamOpen(100,300,10,null,false);
        eq(stream.feed(pcm),1);stream.setLiveDecode(true);eq(stream.getText().text,"ok");
        int calls=Probe.INSTANCE.probe_calls();owner.close();eq(Probe.INSTANCE.probe_calls(),calls+2);
        reject(IllegalStateException.class,()->stream.feed(pcm));stream.close();
        System.out.println("PASS Java ABI: "+assertions+" assertions");
    }
}
'''
PROGRAM_AUX=r'''
import io.github.ggerganov.whispercpp.*;
import com.sun.jna.*;
import java.util.*;
import java.nio.charset.StandardCharsets;
public class WrapperAuxProbe {
    public interface Probe extends Library {
        Probe INSTANCE=Native.load("crispasr",Probe.class);
        void probe_db_mode(int n);int probe_db_live();int probe_db_calls();void probe_titanet_mode(int n);int probe_titanet_live();void probe_aux_mode(int n);int probe_aux_live();int probe_aux_segment_size();int probe_aux_options_size();
    }
    static int assertions;
    static void eq(Object a,Object b){if(!Objects.equals(a,b))throw new AssertionError(a+" != "+b+" expected "+b+" got "+a);assertions++;}
    static void eqFloat(float a,float b){if(Math.abs(a-b)>1e-6)throw new AssertionError(a+" != "+b);assertions++;}
    static void reject(Class<? extends Throwable> cls,Runnable fn) {
        try {fn.run();}catch(Throwable t){if(cls.isInstance(t)){assertions++;return;}throw new AssertionError("expected "+cls.getName()+" got "+t,t);}
        throw new AssertionError("expected "+cls.getName());
    }
    public static void main(String[] args) {
        // pointer width / layout assertions (literal runtime structs)
        eq(24, Probe.INSTANCE.probe_aux_segment_size());
        eq(48, Probe.INSTANCE.probe_aux_options_size());
        // 64-bit pointer width check
        if (Native.POINTER_SIZE != 8) throw new AssertionError("expected 64-bit pointer size");
        assertions++;

        // v2 session open options
        CrispasrAuxiliary.SessionOptions opts=new CrispasrAuxiliary.SessionOptions().threads(7).useGpu(false).verbosity(2).flashAttention(false).gpuLayers(3);
        try(CrispasrSession s=CrispasrAuxiliary.openWithOptions("probe","probe",opts)){
            eq(s.backend(),"probe");
        }
        // malformed open options are implicit via native null return - not testing invalid threads here

        // registry
        CrispasrAuxiliary.RegistryEntry e=CrispasrAuxiliary.registryLookup("foo");
        eq(e.filename,"model.gguf");eq(e.url,"https://test/model");eq(e.size,"1MB");
        eq(CrispasrAuxiliary.registryLookup("missing"),null);
        eq(CrispasrAuxiliary.registryLookupByFilename("model.gguf").filename,"model.gguf");
        String[] backs=CrispasrAuxiliary.registryListBackends();
        eq(backs.length,2);eq(backs[0],"foo");eq(backs[1],"bar");
        CrispasrAuxiliary.RegistryBundle bundle=CrispasrAuxiliary.registryDefaultBundle("foo");
        eq(bundle.backend,"foo");eq(bundle.license,"Apache-2.0");eq(bundle.requiresAcceptance,false);
        eq(bundle.artifacts.length,1);eq(bundle.artifacts[0].kind,1);eq(bundle.artifacts[0].entry.filename,"model.gguf");
        eq(CrispasrAuxiliary.cacheDir(null),"cache");
        eq(CrispasrAuxiliary.cacheEnsureFile("model.gguf","https://test/model",true,null),"cache/model.gguf");
        // positive failure propagation
        Probe.INSTANCE.probe_aux_mode(2);
        reject(IllegalStateException.class, ()-> CrispasrAuxiliary.cacheDir(null));
        reject(IllegalStateException.class, ()-> CrispasrAuxiliary.cacheEnsureFile("model.gguf","https://test/model",true,null));
        Probe.INSTANCE.probe_aux_mode(0);
        reject(NullPointerException.class, ()-> CrispasrAuxiliary.registryLookup(null));
        reject(NullPointerException.class, ()-> CrispasrAuxiliary.registryLookupByFilename(null));
        reject(NullPointerException.class, ()-> CrispasrAuxiliary.registryDefaultBundle(null));

        // speaker embedder lifecycle/dim/name/embed
        float[] pcm={1,3,5};
        try(CrispasrAuxiliary.SpeakerEmbedder emb=new CrispasrAuxiliary.SpeakerEmbedder("probe")){
            eq(emb.dim(),3);eq(emb.name(),"embed-probe");
            float[] out=emb.embed(pcm);
            eqFloat(out[2],0.75f);
            out[2]=999f; float[] out2=emb.embed(pcm); eqFloat(out2[2],0.75f); // copy test
            Probe.INSTANCE.probe_aux_mode(2); reject(IllegalStateException.class, ()-> emb.embed(pcm)); Probe.INSTANCE.probe_aux_mode(0);
            for(int invalid:new int[]{15,16,17}){Probe.INSTANCE.probe_aux_mode(invalid);reject(IllegalStateException.class,()->emb.embed(pcm));}Probe.INSTANCE.probe_aux_mode(0);
        }
        CrispasrAuxiliary.SpeakerEmbedder emb2=new CrispasrAuxiliary.SpeakerEmbedder("probe");
        emb2.close();
        reject(IllegalStateException.class, ()-> emb2.dim());
        reject(IllegalStateException.class, ()-> emb2.embed(pcm));
        reject(IllegalStateException.class, ()-> emb2.name());
        emb2.close(); // double close safe
        reject(NullPointerException.class, ()-> new CrispasrAuxiliary.SpeakerEmbedder(null));
        reject(IllegalArgumentException.class, ()-> new CrispasrAuxiliary.SpeakerEmbedder("",4,null));
        reject(IllegalArgumentException.class, ()-> new CrispasrAuxiliary.SpeakerEmbedder("probe",0,null));
        reject(NullPointerException.class, ()-> { try(CrispasrAuxiliary.SpeakerEmbedder x=new CrispasrAuxiliary.SpeakerEmbedder("probe")){ x.embed(null); } });

        reject(IllegalArgumentException.class, ()->new CrispasrAuxiliary.SpeakerEmbedder("probe\u0000bad"));
        reject(IllegalArgumentException.class, ()->new CrispasrAuxiliary.SpeakerEmbedder("probe",4,"cache\u0000bad"));
        for(int invalid:new int[]{11,12,13}){Probe.INSTANCE.probe_aux_mode(invalid);reject(IllegalStateException.class,()->CrispasrAuxiliary.cluster(new float[]{1,0,0,0,1,0},3,.65f,8));}
        Probe.INSTANCE.probe_aux_mode(14);CrispasrAuxiliary.ClusterResult unassigned=CrispasrAuxiliary.cluster(new float[]{1,0,0,0,1,0},3,.65f,8);eq(unassigned.clusterCount,0);eq(unassigned.labels[1],-1);Probe.INSTANCE.probe_aux_mode(0);
        reject(IllegalArgumentException.class, ()->new CrispasrAuxiliary.PyannoteCache(pcm,"model\u0000bad"));
        // clustering
        CrispasrAuxiliary.ClusterResult cr=CrispasrAuxiliary.cluster(new float[]{1,0,0,0,1,0},3,0.65f,8);
        eq(cr.clusterCount,2);eq(cr.labels[1],1);
        // copy check: mutating returned labels doesn't affect native
        cr.labels[0]=99; CrispasrAuxiliary.ClusterResult cr2=CrispasrAuxiliary.cluster(new float[]{1,0,0,0,1,0},3,0.65f,8); eq(cr2.labels[0],0);
        reject(IllegalArgumentException.class, ()-> CrispasrAuxiliary.cluster(new float[5],3,0.65f,8));
        reject(IllegalArgumentException.class, ()-> CrispasrAuxiliary.cluster(new float[]{1,2},0,0.65f,8));
        reject(NullPointerException.class, ()-> CrispasrAuxiliary.cluster(null,3,0.65f,8));
        reject(IllegalArgumentException.class, ()-> CrispasrAuxiliary.cluster(new float[]{1,2,3},3,Float.NaN,8));

        // pyannote cache compute/apply/free
        CrispasrAuxiliary.DiarizationSegment[] segs={new CrispasrAuxiliary.DiarizationSegment(100,200)};
        try(CrispasrAuxiliary.PyannoteCache cache=new CrispasrAuxiliary.PyannoteCache(pcm,"model")){
            CrispasrAuxiliary.DiarizationSegment[] res=cache.apply(segs,250);
            eq(res[0].speaker,2); eq(res[0].t0,100L);
            eq(segs[0].speaker,-1); // original not mutated -> copy test
            // mutate returned and re-apply should give same
            res[0]=new CrispasrAuxiliary.DiarizationSegment(0,0,99);
            CrispasrAuxiliary.DiarizationSegment[] res2=cache.apply(segs,250); eq(res2[0].speaker,2);
            cache.close();
            reject(IllegalStateException.class, ()-> cache.apply(segs,250));
        }
        try(CrispasrAuxiliary.PyannoteCache c2=new CrispasrAuxiliary.PyannoteCache(pcm,"model")){
            reject(IllegalArgumentException.class, ()-> c2.apply(new CrispasrAuxiliary.DiarizationSegment[0],250));
            reject(NullPointerException.class, ()-> c2.apply(null,250));
        }
        reject(NullPointerException.class, ()-> new CrispasrAuxiliary.PyannoteCache(null,"model"));
        reject(NullPointerException.class, ()-> new CrispasrAuxiliary.PyannoteCache(pcm,null));
        reject(IllegalArgumentException.class, ()-> new CrispasrAuxiliary.PyannoteCache(new float[0],"model"));

        // diarize segments+turns options
        CrispasrAuxiliary.DiarizationOptions dopts=new CrispasrAuxiliary.DiarizationOptions().method(4).nThreads(7).sliceOffsetCs(125).pyannoteModelPath("model").foxNoseEmbedderPath("embed").minSpeakers(1).maxSpeakers(3).numSpeakers(2);
        CrispasrAuxiliary.DiarizationSegment[] inSegs={new CrispasrAuxiliary.DiarizationSegment(100,200)};
        CrispasrAuxiliary.DiarizationResult dr=CrispasrAuxiliary.diarize(pcm,null,inSegs,dopts);
        eq(dr.segments[0].speaker,1); eq(dr.turns[0].t0,125L); eq(dr.turns[0].t1,250L); eq(dr.turns[0].speaker,1);
        eq(inSegs[0].speaker,-1); // copy test
        // without turns
        CrispasrAuxiliary.DiarizationResult dr2=CrispasrAuxiliary.diarizeWithoutTurns(pcm,inSegs,dopts);
        eq(dr2.turns.length,0); eq(dr2.segments[0].speaker,1);
        // malformed
        reject(IllegalArgumentException.class, ()-> CrispasrAuxiliary.diarize(pcm,new float[1],inSegs,dopts));
        reject(IllegalArgumentException.class, ()-> CrispasrAuxiliary.diarize(pcm,null,new CrispasrAuxiliary.DiarizationSegment[0],dopts));
        CrispasrAuxiliary.DiarizationOptions badOpts=new CrispasrAuxiliary.DiarizationOptions().method(99);
        reject(IllegalArgumentException.class, ()-> CrispasrAuxiliary.diarize(pcm,null,inSegs,badOpts));
        reject(NullPointerException.class, ()-> CrispasrAuxiliary.diarize(null,null,inSegs,dopts));
        reject(NullPointerException.class, ()-> CrispasrAuxiliary.diarize(pcm,null,null,dopts));
        reject(NullPointerException.class, ()-> CrispasrAuxiliary.diarize(pcm,null,inSegs,null));
        reject(IllegalArgumentException.class, ()-> CrispasrAuxiliary.diarize(pcm,null,new CrispasrAuxiliary.DiarizationSegment[]{new CrispasrAuxiliary.DiarizationSegment(-1,10)},dopts));
        reject(IllegalArgumentException.class, ()-> CrispasrAuxiliary.diarize(new float[0],null,inSegs,dopts));

        // text language detection
        CrispasrAuxiliary.LanguageDetection ld=CrispasrAuxiliary.detectTextLanguage("test","model");
        eq(ld.label,"de"); eqFloat(ld.confidence,0.875f);
        // copy test: label string is root retained but Java returns copy, mutating not possible but check second call same
        CrispasrAuxiliary.LanguageDetection ld2=CrispasrAuxiliary.detectTextLanguage("test","model"); eq(ld2.label,"de");
        Probe.INSTANCE.probe_aux_mode(2); reject(IllegalStateException.class, ()-> CrispasrAuxiliary.detectTextLanguage("test","model")); Probe.INSTANCE.probe_aux_mode(0);
        reject(NullPointerException.class, ()-> CrispasrAuxiliary.detectTextLanguage(null,"model"));
        reject(NullPointerException.class, ()-> CrispasrAuxiliary.detectTextLanguage("test",null));

        // VAD slices (seconds)
        CrispasrSession.VADSpan[] spans=CrispasrAuxiliary.vadSlices("vad",pcm,16000,0.5f,250,100,30,0f,4);
        eq(spans.length,1); eq(spans[0].t0,1.25); eq(spans[0].t1,2.5);
        // copy test
        double beforeT0=spans[0].t0; spans[0]=null; CrispasrSession.VADSpan[] spans2=CrispasrAuxiliary.vadSlices("vad",pcm); eq(spans2[0].t0,1.25);
        Probe.INSTANCE.probe_aux_mode(1); eq(CrispasrAuxiliary.vadSlices("vad",pcm).length,0); Probe.INSTANCE.probe_aux_mode(0);
        Probe.INSTANCE.probe_aux_mode(2); reject(IllegalStateException.class, ()-> CrispasrAuxiliary.vadSlices("vad",pcm)); Probe.INSTANCE.probe_aux_mode(0);
        reject(NullPointerException.class, ()-> CrispasrAuxiliary.vadSlices(null,pcm));
        reject(IllegalArgumentException.class, ()-> CrispasrAuxiliary.vadSlices("vad",pcm,0,0.5f,250,100,30,0f,4));
        reject(NullPointerException.class, ()-> CrispasrAuxiliary.vadSlices("vad",null));
        eq(CrispasrAuxiliary.vadSlices("vad",new float[0]).length,0);

        // lcs
        eq(CrispasrAuxiliary.lcsDedupPrefixCount(new int[]{1,2},new int[]{2,3},3),1);
        reject(NullPointerException.class, ()-> CrispasrAuxiliary.lcsDedupPrefixCount(null,new int[]{1},3));
        reject(IllegalArgumentException.class, ()-> CrispasrAuxiliary.lcsDedupPrefixCount(new int[]{1},new int[]{1},0));

        CrispasrAuxiliary.Parakeet pk=new CrispasrAuxiliary.Parakeet("probe",4,true);
        CrispasrAuxiliary.ParakeetTranscript pr=pk.transcribe(new float[]{1,3,5},123);
        eq(pr.text,"日本語");eq(pr.words[0].t0,223L);eq(pr.tokens[0].t0,-1L);eq(pr.tokens[0].rawText[0],(byte)0xe6);
        reject(IllegalArgumentException.class,()->pk.transcribe(new float[0],0));pk.close();pk.close();reject(IllegalStateException.class,()->pk.transcribe(new float[]{1},0));
        CrispasrAuxiliary.TitaNet titan=new CrispasrAuxiliary.TitaNet("probe",4);float[] embedding=titan.embed(new float[]{1,3,5});eq(embedding.length,192);eqFloat(embedding[0],1);
        for(int mode:new int[]{2,3,4}){Probe.INSTANCE.probe_titanet_mode(mode);reject(IllegalStateException.class,()->titan.embed(new float[]{1,3,5}));}Probe.INSTANCE.probe_titanet_mode(0);
        reject(IllegalArgumentException.class,()->titan.embed(new float[0]));titan.close();titan.close();reject(IllegalStateException.class,()->titan.embed(new float[]{1,3,5}));eq(Probe.INSTANCE.probe_titanet_live(),0);
        // Usable closed-roster speaker DB API, including native negative no-match scores.
        float[] dbEmbedding={1,3,5};int dbCalls=Probe.INSTANCE.probe_db_calls();
        reject(IllegalArgumentException.class,()->new CrispasrAuxiliary.SpeakerDb("db","Alice,Bob",false));
        reject(IllegalArgumentException.class,()->new CrispasrAuxiliary.SpeakerDb("db","",true));eq(Probe.INSTANCE.probe_db_calls(),dbCalls);
        CrispasrAuxiliary.SpeakerDb db=new CrispasrAuxiliary.SpeakerDb("db","Alice,Bob",true);
        eq(db.count(),2);eq(Probe.INSTANCE.probe_db_live(),1);
        CrispasrAuxiliary.SpeakerMatch match=db.match(dbEmbedding);eq(match.name,"アリス");eqFloat(match.score,.875f);
        Probe.INSTANCE.probe_db_mode(1);eq(db.match(dbEmbedding).name,null);eqFloat(db.match(dbEmbedding).score,-1f);
        eq(db.match(dbEmbedding,-1f).name,null);eq(match.name,"アリス");Probe.INSTANCE.probe_db_mode(0);
        reject(IllegalArgumentException.class,()->db.match(new float[0]));reject(NullPointerException.class,()->db.match(null));
        reject(IllegalArgumentException.class,()->db.match(dbEmbedding,Float.NaN));
        reject(IllegalArgumentException.class,()->CrispasrAuxiliary.SpeakerDb.enroll("db","Alice",dbEmbedding,false));
        reject(IllegalArgumentException.class,()->CrispasrAuxiliary.SpeakerDb.enroll("db","Alice",new float[0],true));
        CrispasrAuxiliary.SpeakerDb.enroll("db","Alice",dbEmbedding,true);assertions++;
        Probe.INSTANCE.probe_db_mode(3);reject(IllegalStateException.class,()->CrispasrAuxiliary.SpeakerDb.enroll("db","Alice",dbEmbedding,true));
        Probe.INSTANCE.probe_db_mode(2);reject(IllegalStateException.class,()->new CrispasrAuxiliary.SpeakerDb("db","Alice,Bob",true));Probe.INSTANCE.probe_db_mode(0);
        db.close();db.close();eq(Probe.INSTANCE.probe_db_live(),0);dbCalls=Probe.INSTANCE.probe_db_calls();
        reject(IllegalStateException.class,()->db.count());reject(IllegalStateException.class,()->db.match(dbEmbedding));
        eq(Probe.INSTANCE.probe_db_calls(),dbCalls);
        // closed handles and buffer cleanup already checked via live
        eq(Probe.INSTANCE.probe_aux_live(),0);
        System.out.println("PASS Java auxiliary: "+assertions+" assertions");
    }
}
'''


@unittest.skipUnless(JAVA and JAR and Path(JAR).is_file(), 'Java runtime/compiler and CRISPASR_JNA_JAR required')
class JavaWrapperContract(unittest.TestCase):
    def test_alignment_worker_compiles_parent_classes_and_utf8_boundary(self):
        # Execute the worker's actual javac argv without importing its model job.
        import ast
        tree = ast.parse((ROOT/'tools/ci-heavy/issue490_characters.py').read_text())
        commands = [node.args[0] for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == 'run' and len(node.args) == 2
                    and isinstance(node.args[1], ast.Constant) and node.args[1].value == 'java-compile']
        self.assertEqual(len(commands), 1)
        with tempfile.TemporaryDirectory(prefix='java-align-preflight-') as tmp:
            path = Path(tmp)
            javac = shutil.which('javac')
            compiler = [javac] if javac else [JAVA, '-m', 'jdk.compiler/com.sun.tools.javac.Main']
            argv = eval(compile(ast.Expression(commands[0]), '<worker javac argv>', 'eval'),
                        {'ROOT': ROOT, 'jna': Path(JAR), 'classes': path})
            subprocess.run(compiler + list(map(str, argv[1:])), cwd=path, check=True)
            subprocess.run([os.environ.get('CC') or 'cc', '-shared', '-fPIC',
                            str(ROOT/'tools/ci-heavy/issue490_jna_utf8.c'),
                            '-o', str(path/'libcrispasr.so')], check=True)
            for encoding in ('US-ASCII', 'ISO-8859-1', 'UTF-8'):
                run = subprocess.run([JAVA, '-Djna.tmpdir='+str(path), '-Djna.library.path='+str(path),
                                      '-Dfile.encoding=UTF-8', '-Djna.encoding='+encoding,
                                      '-cp', str(path)+os.pathsep+JAR, 'Issue490Utf8'],
                                     capture_output=True, text=True, timeout=30)
                self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
                self.assertIn('JAVA_BIDIRECTIONAL_UTF8_PASS', run.stdout)

    def test_transport_ownership_metadata_and_task_operations(self):
        with tempfile.TemporaryDirectory(prefix='java-contract-') as tmp:
            path=Path(tmp);(path/'WrapperAbiProbe.java').write_text(PROGRAM)
            src=(ROOT/'src/crispasr_c_api.cpp').read_text()
            start=src.index('struct crispasr_vad_abi_opts {')
            (path/'runtime-vad-layout.h').write_text(src[start:src.index('};',start)+2])
            sources=[ROOT/'bindings/java/src/main/java/io/github/ggerganov/whispercpp'/name
                     for name in ('CrispasrSession.java','CrispasrSessionTasks.java','CrispasrUtilities.java')]
            javac=shutil.which('javac')
            compiler=[javac] if javac else [JAVA,'-m','jdk.compiler/com.sun.tools.javac.Main']
            subprocess.run(compiler+['-encoding','UTF-8','-cp',JAR,'-d',str(path),*map(str,sources),str(path/'WrapperAbiProbe.java')],check=True)
            for old in (False,True):
                directory=path/('old' if old else 'current');directory.mkdir()
                cc=os.environ.get('CC') or 'cc'
                subprocess.run([cc,'-std=c11','-shared','-fPIC','-Werror',
                                '-I'+str(ROOT/'include'), '-I'+str(ROOT/'ggml/include'), '-I'+str(path), '-DCRISPASR_PROBE_JAVA_RNNOISE', '-DCRISPASR_PROBE_VAD_ARGUMENTS', '-DCRISPASR_PROBE_EXISTING_STREAM_OPS', *(['-DCRISPASR_PROBE_OLD_TOKENS'] if old else []),
                                str(ROOT/'tests/fixtures/python-wrapper-abi.c'),str(ROOT/'tests/fixtures/wrapper-session-metadata.c'),str(ROOT/'tests/fixtures/wrapper-stream-lifetime.c'),str(ROOT/'tests/fixtures/wrapper-audio-utilities.c'),
                                '-o',str(directory/'libcrispasr.so')],check=True)
                run=subprocess.run([JAVA,'-Djna.tmpdir='+str(path),'-Djna.library.path='+str(directory),
                                    '-cp',str(path)+os.pathsep+JAR,'WrapperAbiProbe',*(['old'] if old else [])],capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr);self.assertIn('PASS Java',run.stdout)
                print(run.stdout.strip())


@unittest.skipUnless(JAVA and JAR and Path(JAR).is_file(), 'Java runtime/compiler and CRISPASR_JNA_JAR required')
class JavaAuxWrapperContract(unittest.TestCase):
    def test_auxiliary_registry_speaker_diarize_vad_and_session_options(self):
        with tempfile.TemporaryDirectory(prefix='java-aux-') as tmp:
            path=Path(tmp)
            (path/'WrapperAuxProbe.java').write_text(PROGRAM_AUX)
            sources=[ROOT/'bindings/java/src/main/java/io/github/ggerganov/whispercpp'/name
                     for name in ('CrispasrSession.java','CrispasrSessionTasks.java','CrispasrUtilities.java','CrispasrAuxiliary.java')]
            javac=shutil.which('javac')
            compiler=[javac] if javac else [JAVA,'-m','jdk.compiler/com.sun.tools.javac.Main']
            subprocess.run(compiler+['-encoding','UTF-8','-cp',JAR,'-d',str(path),*map(str,sources),str(path/'WrapperAuxProbe.java')],check=True)
            # Extract literal POD layouts from src/crispasr_c_api.cpp
            src=(ROOT/'src/crispasr_c_api.cpp').read_text()
            layouts=[]
            for name in ['crispasr_diarize_seg_abi','crispasr_diarize_turn_abi','crispasr_diarize_opts_abi']:
                start=src.index('struct '+name+' {')
                end=src.index('};',start)+2
                layouts.append(src[start:end])
            (path/'runtime-abi-layout.h').write_text('\n'.join(layouts))
            directory=path/'current';directory.mkdir()
            cc2=os.environ.get('CC') or 'cc'
            subprocess.run([cc2,'-std=c11','-shared','-fPIC','-Werror',
                            '-I'+str(ROOT/'include'), '-I'+str(ROOT/'ggml/include'), '-I'+str(path),
                            str(ROOT/'tests/fixtures/python-wrapper-abi.c'),
                            str(ROOT/'tests/fixtures/wrapper-session-metadata.c'),
                            str(ROOT/'tests/fixtures/csharp-wrapper-aux.c'),str(ROOT/'tests/fixtures/wrapper-parakeet-legacy.c'),str(ROOT/'tests/fixtures/wrapper-titanet-legacy.c'),str(ROOT/'tests/fixtures/wrapper-speaker-db.c'),
                            '-o',str(directory/'libcrispasr.so')],check=True)
            run=subprocess.run([JAVA,'-Djna.tmpdir='+str(path),'-Djna.library.path='+str(directory),
                                '-cp',str(path)+os.pathsep+JAR,'WrapperAuxProbe'],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('PASS Java auxiliary:',run.stdout)
            print(run.stdout.strip())


@unittest.skipUnless(JAVA and JAR and Path(JAR).is_file(), 'Java runtime/compiler and CRISPASR_JNA_JAR required')
class JavaLegacyWrapperContract(unittest.TestCase):
    def test_legacy_context_options_owned_parameters_and_repeated_close(self):
        from build_legacy_wrapper_probe import build
        with tempfile.TemporaryDirectory(prefix='java-legacy-') as tmp:
            path=Path(tmp);build(path/'libcrispasr.so')
            sources=list((ROOT/'bindings/java/src/main/java').rglob('*.java'))
            javac=shutil.which('javac')
            compiler=[javac] if javac else [JAVA,'-m','jdk.compiler/com.sun.tools.javac.Main']
            subprocess.run(compiler+['-encoding','UTF-8','-cp',JAR,'-d',str(path),*map(str,sources),str(ROOT/'tests/fixtures/JavaLegacyProbe.java')],check=True)
            for encoding in (None,'ISO-8859-1'):
                run=subprocess.run([JAVA,'-Djna.tmpdir='+str(path),'-Djna.library.path='+str(path),
                    *(['-Djna.encoding='+encoding] if encoding else []),'-cp',str(path)+os.pathsep+JAR,'JavaLegacyProbe'],
                    capture_output=True,text=True,timeout=30)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                self.assertIn('PASS Java legacy:',run.stdout);print(run.stdout.strip())


if __name__=='__main__':unittest.main()
