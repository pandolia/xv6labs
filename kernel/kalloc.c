// Physical memory allocator, for user processes,
// kernel stacks, page-table pages,
// and pipe buffers. Allocates whole 4096-byte pages.

#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "spinlock.h"
#include "riscv.h"
#include "defs.h"

void freerange(void *pa_start, void *pa_end);

extern char end[]; // first address after kernel.
                   // defined by kernel.ld.

struct run {
  struct run *next;
};

struct {
  struct spinlock lock;
  struct run *freelist;
} kmems[NCPU];

void
kinit()
{
  for(int i=0; i<NCPU; i++)
    initlock(&kmems[i].lock, "kmem");
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

  // Fill with junk to catch dangling refs.
  memset(pa, 1, PGSIZE);

  r = (struct run*)pa;

  /** 获取 cpuid 时要关中断，完事之后再把中断开下来 */
  push_off();
  int id = cpuid();
  pop_off();

  /** 将空闲的 page 归还给第 id 个 CPU */
  acquire(&kmems[id].lock);
  r->next = kmems[id].freelist;
  kmems[id].freelist = r;
  release(&kmems[id].lock);
}

// Allocate one 4096-byte page of physical memory.
// Returns a pointer that the kernel can use.
// Returns 0 if the memory cannot be allocated.
// Allocate one 4096-byte page of physical memory.
// Returns a pointer that the kernel can use.
// Returns 0 if the memory cannot be allocated.
void *
kalloc(void)
{
  struct run *r;

  /** 获取 cpuid 时要关中断，完事之后再把中断开下来 */
  push_off();
  int id = cpuid();
  pop_off();

  /** 尝试获取剩余的空闲 page */
  acquire(&kmems[id].lock);
  r = kmems[id].freelist;
  if(r) {
    kmems[id].freelist = r->next;
  } else {
    for(int i=0; i<NCPU; i++) {
      if(i == id)
        continue;

      /** 尝试偷一个其他 CPU 的空闲 page */
      acquire(&kmems[i].lock);
      if(!kmems[i].freelist) {
        release(&kmems[i].lock);
        continue;
      }
      
      r = kmems[i].freelist;
      kmems[i].freelist = r->next;
      release(&kmems[i].lock);
      break;
    }
  }
  release(&kmems[id].lock);

  /** 有一种可能：第id个CPU没有空闲 page ，也没偷到 */
  if(r)
    memset((char*)r, 5, PGSIZE); // fill with junk
  return (void*)r;
}

