import os
from subprocess import check_output

from ..sysutils import real_path, get_temp_file
from .metric import Metric

class METEORScore(Metric):
    def __init__(self, score=None):
        super(METEORScore, self).__init__(score)
        self.name = "METEOR"
        self.score = (100*score) if score else 0.
        self.score_str = "%.5f" % self.score

class METEORScorer(object):
    def __init__(self):
        self.path = real_path(os.environ.get('METEOR_JAR', None))
        if self.path is None or not os.path.exists(self.path):
            raise Exception("METEOR jar file not found.")

        self.__cmdline = ["java", "-Xmx2G", "-jar", self.path]

    def compute(self, refs, hyps, language="auto", norm=True):
        cmdline = self.__cmdline[:]

        if isinstance(hyps, list):
            # Create a temporary file
            with get_temp_file(suffix=".hyps") as tmpf:
                for hyp in hyps:
                    tmpf.write("%s\n" % hyp)

                cmdline.append(tmpf.name)

        elif isinstance(hyps, str):
            cmdline.append(hyps)

        # Make reference files a list
        refs = [refs] if isinstance(refs, str) else refs
        n_refs = len(refs)
        if n_refs > 1:
            # Multiple references
            # FIXME: METEOR can consume everything from stdin
            tmpff = get_temp_file(suffix=".refs")
            fname = tmpff.name
            tmpff.close()
            os.system('paste -d"\\n" %s > %s' % (" ".join(refs), fname))
            cmdline.append(fname)
        else:
            cmdline.append(refs[0])

        if language == "auto":
            # Take the extension of the 1st reference file, e.g. ".de"
            language = os.path.splitext(refs[0])[-1][1:]

        cmdline.extend(["-l", language])
        if norm:
            cmdline.append("-norm")

        if n_refs > 1:
            # Multiple references
            cmdline.extend(["-r", str(n_refs)])

        output = check_output(cmdline)

        score = output.splitlines()
        if len(score) == 0:
            return METEORScore()
        else:
            # Final score: 0.320320320320
            return METEORScore(float(score[-1].split(":")[-1].strip()))
