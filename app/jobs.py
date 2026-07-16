"""Minimal background job queue (one GPU worker) with progress tracking."""
import queue
import threading
import time
import traceback
import uuid


class Job:
    def __init__(self, kind):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.status = "queued"
        self.progress = 0.0
        self.message = "Queued"
        self.results = []       # list of {"file": name, "label": str}
        self.inputs = []        # list of input image filenames
        self.error = None
        self.created = time.time()

    def set_progress(self, p, message=None):
        self.progress = float(p)
        if message:
            self.message = message

    def to_dict(self):
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "results": self.results,
            "inputs": self.inputs,
            "error": self.error,
        }


class JobManager:
    def __init__(self):
        self.jobs = {}
        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def submit(self, kind, fn):
        """fn(job) runs on the worker thread."""
        job = Job(kind)
        self.jobs[job.id] = job
        self._queue.put((job, fn))
        return job

    def get(self, job_id):
        return self.jobs.get(job_id)

    def _loop(self):
        while True:
            job, fn = self._queue.get()
            job.status = "running"
            job.message = "Starting"
            try:
                fn(job)
                job.status = "done"
                job.progress = 1.0
                job.message = "Completed"
            except Exception as e:  # surface errors to the frontend
                traceback.print_exc()
                job.status = "error"
                job.error = str(e)
                job.message = f"Failed: {e}"
