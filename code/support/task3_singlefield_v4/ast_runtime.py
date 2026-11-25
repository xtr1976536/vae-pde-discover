"""GRE-free explicit mathematical AST and coefficient-fitting runtime.
All names are declared by the task; LaTeX is parsed, never executed as Python.
Coordinates use single-character names, with arbitrary declared spatial dimension.
"""
import re
import numpy as np
class LatexError(ValueError): pass
class Unsupported(LatexError): pass
FUNCS={'sin':np.sin,'cos':np.cos,'tan':np.tan,'sinh':np.sinh,'cosh':np.cosh,'tanh':np.tanh,'exp':np.exp,'log':np.log,'ln':np.log,'sqrt':np.sqrt,'abs':np.abs}
NUMBER=r'(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?'
class LatexParser:
    def __init__(self,text,fields=('u','v','p'),coordinates=('x','y','z','t'),parameters=(),functions=()):
        self.fields=set(fields);self.coordinates=set(coordinates);self.parameters=set(parameters);self.functions=set(functions)
        if self.fields & (self.coordinates|self.parameters) or self.parameters & self.coordinates:raise LatexError('Names have conflicting roles')
        if any(len(a)!=1 for a in coordinates):raise Unsupported('Use single-character coordinate names')
        if len(text)>16384:raise LatexError('Input too long')
        names=sorted(self.fields|self.coordinates|self.parameters|self.functions,key=len,reverse=True)
        name_pattern='|'.join(re.escape(s) for s in names) or r'(?!)'
        token=re.compile(NUMBER+'|'+name_pattern+r'|\\[A-Za-z]+|\\[^A-Za-z]|[^\s]')
        self.tokens=[m.group() for m in token.finditer(text) if m.group() not in (r'\left',r'\right',r'\,',r'\!',r'\;',r'\displaystyle')];self.p=0
    def peek(self):return self.tokens[self.p] if self.p<len(self.tokens) else None
    def take(self,expected=None):
        t=self.peek()
        if t is None:raise LatexError('Unexpected end')
        if expected is not None and expected!=t:raise LatexError('Expected '+expected+', got '+t)
        self.p+=1;return t
    def group(self):
        opening=self.take();closing={'{':'}','(':')','[':']'}.get(opening)
        if closing is None:raise LatexError('A group is required')
        a=self.expression();self.take(closing);return a
    def exponent(self):
        if self.peek() in ('{','('):return self.group()
        if self.peek() in ('+','-'):
            sign=self.take();v=self.atom(scripts=False);return ('neg',v) if sign=='-' else v
        return self.atom(scripts=False)
    def integer_exponent(self):
        n=self.exponent()
        if n[0]!='n' or n[1]!=int(n[1]):raise Unsupported('Derivative order must be a positive integer')
        v=int(n[1])
        if not 1<=v<=8:raise Unsupported('Derivative order must be 1..8')
        return v
    def derivative_variables(self):
        grouped=self.peek()=='{'
        if grouped:self.take()
        axes=''
        while self.peek() in self.coordinates:
            v=self.take();repeat=1
            if self.peek()=='^':self.take();repeat=self.integer_exponent()
            axes+=v*repeat
            if not grouped:break
        if grouped:self.take('}')
        if not axes or len(axes)>8:raise LatexError('Derivative coordinates required (max order 8)')
        return axes
    def fraction(self):
        self.take() # frac/dfrac/tfrac
        if self.peek()=='{' and self.p+1<len(self.tokens) and self.tokens[self.p+1]==r'\partial':
            self.take('{');self.take(r'\partial');order=1
            if self.peek()=='^':self.take();order=self.integer_exponent()
            field=self.expression();self.take('}');self.take('{');axes=''
            while self.peek()!='}':
                self.take(r'\partial')
                if self.peek() not in self.coordinates:raise LatexError('Unknown derivative coordinate')
                a=self.take();repeat=1
                if self.peek()=='^':self.take();repeat=self.integer_exponent()
                axes+=a*repeat
            self.take('}')
            if len(axes)!=order:raise LatexError('Numerator/denominator derivative order mismatch')
            return ('d',field,axes)
        return ('div',self.group(),self.group())
    def atom(self,scripts=True):
        t=self.peek()
        if t is None:raise LatexError('Unexpected end')
        if t in ('{','(','['):a=self.group()
        elif t in self.fields|self.coordinates|self.parameters:
            name=self.take();a=('v',name)
            # A field evaluated at the task coordinates is the same field sample.
            if name in self.fields and self.peek()=='(':
                saved=self.p;self.take();coords=[]
                while self.peek() in self.coordinates:
                    coords.append(self.take())
                    if self.peek()!=',':break
                    self.take(',')
                if self.peek()==')' and coords:self.take(')')
                else:self.p=saved
        elif re.fullmatch(NUMBER,t):a=('n',float(self.take()))
        elif t in (r'\frac',r'\dfrac',r'\tfrac'):a=self.fraction()
        elif t.startswith('\\') and t[1:] in FUNCS:
            self.take();a=('fn',t[1:],self.group() if self.peek() in ('{','(','[') else self.atom())
        elif t in self.functions:
            name=self.take();self.take('(');args=[self.expression()]
            while self.peek()==',':self.take();args.append(self.expression())
            self.take(')');a=('call',name,*args)
        elif t==r'\partial':
            self.take();preorder=1
            if self.peek()=='^':self.take();preorder=self.integer_exponent()
            self.take('_');axes=self.derivative_variables()
            if self.peek()=='^':self.take();axes*=self.integer_exponent()
            if preorder>1:
                if len(axes)!=1:raise LatexError('Ambiguous derivative order')
                axes*=preorder
            operand=self.group() if self.peek() in ('{','(','[') else self.atom()
            a=('d',operand,axes)
        elif t==r'\Delta':
            self.take();order=1
            if self.peek()=='^':self.take();order=self.integer_exponent()
            a=('lap',self.group() if self.peek() in ('{','(','[') else self.atom(),order)
        elif t==r'\nabla':
            self.take()
            if self.peek()==r'\cdot':self.take();kind='divergence'
            else:kind='grad'
            a=(kind,self.group() if self.peek() in ('{','(','[') else self.atom())
        elif t in ('-','+'):
            self.take();v=self.atom();a=('neg',v) if t=='-' else v
        elif t=='|':
            self.take();v=self.expression();self.take('|');a=('fn','abs',v)
        else:raise Unsupported('Undeclared/unsupported LaTeX token: '+str(t))
        if scripts and self.peek()=='_':
            if a[0]!='v' or a[1] not in self.fields:raise Unsupported('Only fields have derivative subscripts')
            self.take();a=('d',a,self.derivative_variables())
        if scripts and self.peek()=='^':self.take();a=('pow',a,self.exponent())
        return a
    def product(self):
        a=self.atom()
        while self.peek() is not None and self.peek() not in ('+','-','=',')','}',']',',','|'):
            explicit=self.peek() in ('*',r'\cdot',r'\times','/')
            op=self.take() if explicit else '*'
            b=self.atom();a=('div',a,b) if op=='/' else ('mul',a,b)
        return a
    def expression(self):
        a=self.product()
        while self.peek() in ('+','-'):
            op=self.take();b=self.product();a=('add',a,b if op=='+' else ('neg',b))
        return a
    def parse(self):
        try:
            a=self.expression()
            if self.peek()=='=':self.take();a=('add',a,('neg',self.expression()))
            if self.peek() is not None:raise LatexError('Trailing input')
            return a
        except RecursionError as e:raise LatexError('Nesting limit') from e

def evaluate(node,values,derivative):
    k=node[0]
    with np.errstate(all='raise'):
        if k=='v':result=np.asarray(values[node[1]])
        elif k=='n':result=float(node[1])
        elif k=='d':result=np.asarray(derivative(node[1],node[2]))
        elif k in ('lap','grad','divergence'):raise Unsupported('Expand spatial operators before evaluation')
        elif k=='neg':result=-evaluate(node[1],values,derivative)
        elif k=='pow':result=evaluate(node[1],values,derivative)**evaluate(node[2],values,derivative)
        elif k=='fn':result=FUNCS[node[1]](evaluate(node[2],values,derivative))
        elif k=='call':result=values[node[1]](*(evaluate(n,values,derivative) for n in node[2:]))
        elif k=='add':result=sum(evaluate(n,values,derivative) for n in node[1:])
        elif k=='mul':result=np.prod(np.broadcast_arrays(*(evaluate(n,values,derivative) for n in node[1:])),axis=0)
        elif k=='div':result=evaluate(node[1],values,derivative)/evaluate(node[2],values,derivative)
        else:raise Unsupported(k)
    if np.iscomplexobj(result):raise Unsupported('Real-valued fields required; split complex fields into components')
    if not np.isfinite(result).all():raise ValueError('Nonfinite expression')
    return result

def expand_laplacian(node,spatial=('x','y')):
    def components(n):
        if n[0]=='grad':return [('d',expand_laplacian(n[1],spatial),a) for a in spatial]
        if n[0]=='mul':
            vector=[i for i,v in enumerate(n[1:]) if v[0] in ('grad',)]
            if len(vector)!=1:raise Unsupported('Divergence operand requires one explicit gradient factor')
            i=vector[0];parts=components(n[i+1]);return [('mul',*(expand_laplacian(v,spatial) if j!=i else p for j,v in enumerate(n[1:]))) for p in parts]
        if n[0]=='add':
            parts=[components(v) for v in n[1:]];return [('add',*(p[i] for p in parts)) for i in range(len(spatial))]
        raise Unsupported('Explicit gradient flux required for divergence')
    if node[0]=='lap':
        out=expand_laplacian(node[1],spatial)
        for _ in range(node[2]):out=('add',*(('d',out,a+a) for a in spatial))
        return out
    if node[0]=='divergence':return ('add',*(('d',v,a) for a,v in zip(spatial,components(node[1]))))
    if node[0]=='grad':raise Unsupported('Bare vector gradient needs explicit scalar component or divergence')
    return tuple(expand_laplacian(v,spatial) if isinstance(v,tuple) else v for v in node)

def terms(node):
    if node[0]=='add':return [t for n in node[1:] for t in terms(n)]
    if node[0]=='neg':return [('neg',t) for t in terms(node[1])]
    return [node]

def fit_score(rhs,target,values,derivative,ridge=0.,train_mask=None,score_mask=None):
    if ridge<0:raise ValueError('negative ridge')
    if np.iscomplexobj(target):raise Unsupported('Real-valued targets required')
    y=np.asarray(target,dtype=float).ravel()
    columns=[np.broadcast_to(evaluate(t,values,derivative),np.shape(target)).ravel() for t in terms(rhs)]
    X=np.column_stack(columns)
    if not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError('Nonfinite candidate or target')
    if train_mask is None or score_mask is None:raise ValueError('Explicit fit and score masks required')
    fit=np.asarray(train_mask,dtype=bool).ravel();score=np.asarray(score_mask,dtype=bool).ravel()
    if fit.size!=y.size or score.size!=y.size or not fit.any() or not score.any() or np.any(fit & score):raise ValueError('Invalid/disjoint masks required')
    A=X[fit];b=y[fit]
    if ridge:A=np.vstack([A,np.sqrt(ridge)*np.eye(X.shape[1])]);b=np.r_[b,np.zeros(X.shape[1])]
    coef=np.linalg.lstsq(A,b,rcond=None)[0];residual=X[score]@coef-y[score]
    mse=float(np.mean(residual**2));relative=mse/max(float(np.mean(y[score]**2)),1e-24)
    if not np.isfinite(relative):raise ValueError('Nonfinite residual')
    return dict(coefficients=coef.tolist(),mse=mse,relative_mse=relative,reward=1/(1+relative),fit_points=int(fit.sum()),score_points=int(score.sum()))

def fit_score_multi(rhs,trajectories,field,ridge=0.):
    """One common coefficient vector, trajectory-balanced fit and scoring.
    Samples are processed trajectory by trajectory; no trajectory encoder is added.
    Fit normalization uses fit targets only. Score residuals use separate masks.
    """
    if not trajectories:raise ValueError('At least one trajectory required')
    matrices=[];targets=[];blocks=[]
    for tr in trajectories:
        y=np.asarray(tr['targets'][field],dtype=float);shape=y.shape;y=y.ravel()
        X=np.column_stack([np.broadcast_to(evaluate(t,tr['values'],tr['derivative']),shape).ravel() for t in terms(rhs)])
        fit=np.asarray(tr['fit_mask'],dtype=bool).ravel();score=np.asarray(tr['score_mask'],dtype=bool).ravel()
        if fit.size!=y.size or score.size!=y.size or not fit.any() or not score.any() or np.any(fit & score):raise ValueError('Invalid/disjoint trajectory masks')
        if not np.isfinite(y).all() or not np.isfinite(X).all():raise ValueError('Nonfinite trajectory data')
        weight=np.sqrt(max(float(np.mean(y[fit]**2)),1e-24)*fit.sum())
        matrices.append(X[fit]/weight);targets.append(y[fit]/weight);blocks.append((X,y,fit,score))
    A=np.vstack(matrices);b=np.concatenate(targets)
    if ridge<0:raise ValueError('negative ridge')
    if ridge:A=np.vstack([A,np.sqrt(ridge)*np.eye(A.shape[1])]);b=np.r_[b,np.zeros(A.shape[1])]
    coefficients=np.linalg.lstsq(A,b,rcond=None)[0];scores=[]
    for X,y,fit,mask in blocks:
        mse=float(np.mean((X[mask]@coefficients-y[mask])**2));relative=mse/max(float(np.mean(y[mask]**2)),1e-24)
        if not np.isfinite(relative):raise ValueError('Nonfinite trajectory score')
        scores.append(dict(mse=mse,relative_mse=relative,fit_points=int(fit.sum()),score_points=int(mask.sum())))
    relative=float(np.mean([s['relative_mse'] for s in scores]))
    return dict(coefficients=coefficients.tolist(),relative_mse=relative,reward=1/(1+relative),trajectories=scores)
