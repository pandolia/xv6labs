// Physical memory allocator, for user processes,
// kernel stacks, page-table pages,
// and pipe buffers. Allocates whole 4096-byte pages.

#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "spinlock.h"
#include "riscv.h"
#include "defs.h"

/** 引用计数块 */
typedef struct {
  struct spinlock lock;
  int count;
} memref;

#define MEMREFS PHYSTOP/PGSIZE

/** 引用计数向量，监测每个页块 */
memref memrefs[MEMREFS];

void freerange(void *pa_start, void *pa_end);

extern char end[]; // first address after kernel.
                   // defined by kernel.ld.

struct run {
  struct run *next;
};

struct {
  struct spinlock lock;
  struct run *freelist;
} kmem;

void
kinit()
{
  /** 初始化引用计数向量 */
  for(int i=0; i<MEMREFS; i++) 
    initlock(&(memrefs[i].lock), "memrefs");

  initlock(&kmem.lock, "kmem");
  freerange(end, (void*)PHYSTOP);
}

void
freerange(void *pa_start, void *pa_end)
{
  char *p;
  p = (char*)PGROUNDUP((uint64)pa_start);
  for(; p + PGSIZE <= (char*)pa_end; p += PGSIZE)
    kfree(p);
}

// Free the page of physical memory pointed at by v,
// which normally should have been returned by a
// call to kalloc().  (The exception is when
// initializing the allocator; see kinit above.)
void
kfree(void *pa)
{
  struct run *r;

  if(((uint64)pa % PGSIZE) != 0 || (char*)pa < end || (uint64)pa >= PHYSTOP)
    panic("kfree");

  /** 释放页块时需要解引用 */
  uint32 i = (uint64)pa/PGSIZE;
  acquire(&(memrefs[i].lock));
  memrefs[i].count--;

  if(memrefs[i].count > 0) 
    goto notzero;

  // Fill with junk to catch dangling refs.
  memset(pa, 1, PGSIZE);

  r = (struct run*)pa;

  acquire(&kmem.lock);
  r->next = kmem.freelist;
  kmem.freelist = r;
  release(&kmem.lock);

notzero:
  release(&(memrefs[i].lock));
}

// Allocate one 4096-byte page of physical memory.
// Returns a pointer that the kernel can use.
// Returns 0 if the memory cannot be allocated.
void *
kalloc(void)
{
  struct run *r;

  acquire(&kmem.lock);
  r = kmem.freelist;
  
  if(r) {
    /** 初始化引用计数，刚alloc完，引用必然为1 */
    uint32 i = (uint64)r/PGSIZE;
    acquire(&(memrefs[i].lock));
    memrefs[i].count = 1;
    release(&(memrefs[i].lock));

    kmem.freelist = r->next;
  }
  release(&kmem.lock);

  if(r)
    memset((char*)r, 5, PGSIZE); // fill with junk
  return (void*)r;
}

/** 新分配的页块，也要增加引用 */
int
incr_ref(void* pa)
{
   if(((uint64)pa%PGSIZE)!=0 || (char*)pa<end || (uint64)pa>=PHYSTOP)
    return -1;

  uint32 i = (uint64)pa/PGSIZE;
  acquire(&(memrefs[i].lock));
  memrefs[i].count++;
  release(&(memrefs[i].lock));
  return 1;
}

/** 检查该页块是否为copy-on-write */
int 
iscow(pagetable_t pagetable, uint64 va)
{
  if(va > MAXVA)
    return 0;

  pte_t* pte = walk(pagetable, va, 0);
  if(pte==0 || (*pte&PTE_V)==0)
    return 0;

  return (*pte&PTE_COW);
}

/** copy-on-write拷贝工作（父->子） */
uint64 
cowcopy(pagetable_t pagetable, uint64 va)
{
  va = PGROUNDDOWN(va);
  pte_t* pte = walk(pagetable, va, 0);
  uint64 pa = PTE2PA(*pte);
  uint32 i = pa/PGSIZE;

  /** 检查是否只有一个进程正在使用该页块 */
  acquire(&(memrefs[i].lock));
  if(memrefs[i].count == 1) {
    /** 恢复该页块的写入权限 */
    *pte |= PTE_W;
    *pte &= (~PTE_COW);
    release(&(memrefs[i].lock));
    return pa;
  }

  release(&(memrefs[i].lock));
  /** 尝试分配空间（缺页中断handler） */
  char* mem = kalloc();
  if(mem == 0)
    return 0;

  /** 拷贝工作 */
  memmove(mem, (char*)pa, PGSIZE);
  *pte &= (~PTE_V);
  uint64 flag = PTE_FLAGS(*pte);
  flag |= PTE_W;
  flag &= (~PTE_COW);

  if(mappages(pagetable, va, PGSIZE, (uint64)mem, flag) != 0) 
    goto freeing;
  
  /** 顺利结束拷贝工作 */
  goto rest;

freeing:
  kfree(mem);
  return 0;
  
rest:
  kfree((char*)PGROUNDDOWN(pa));
  return (uint64)mem;
}

